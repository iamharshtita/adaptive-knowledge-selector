"""
S3 Sync Utility for Team Collaboration

Syncs all large/computed assets to S3:
- Model weights (adaptive_dqn.pth)
- Knowledge graph (hetionet_graph.pkl)
- PDF vector store (pdf_store/)
- Training datasets

Usage:
    # Sync everything
    python -m utils.s3_sync sync-all

    # Sync specific components
    python -m utils.s3_sync download-kg
    python -m utils.s3_sync upload-model -m "Training message"
    python -m utils.s3_sync download-pdf-store

    # Check status
    python -m utils.s3_sync status
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from datetime import datetime
import json
import hashlib
from pathlib import Path
import shutil


class S3TeamSync:
    """Manages S3 synchronization for all team-shared assets."""

    # Define all assets and their S3 locations
    ASSETS = {
        'model': {
            'local': 'data/rl_selector/adaptive_dqn.pth',
            's3_key': 'models/adaptive_dqn.pth',
            'description': 'RL model weights',
            'versioned': True  # Track changes frequently
        },
        'kg': {
            'local': 'data/hetionet/hetionet_graph.pkl',
            's3_key': 'knowledge_graph/hetionet_graph.pkl',
            'description': 'Preprocessed Hetionet knowledge graph',
            'versioned': False  # Static after initial processing
        },
        'pdf_store': {
            'local': 'data/pdf_store/',
            's3_key': 'pdf_store/',
            'description': 'PDF vector store (FAISS index + metadata)',
            'versioned': False  # Static after initial ingestion
        },
        'training_data': {
            'local': 'data/training_dataset_600.json',
            's3_key': 'datasets/training_dataset_600.json',
            'description': '600-query training dataset',
            'versioned': True  # May be updated
        },
        'training_log': {
            'local': 'data/rl_selector/training_log.json',
            's3_key': 'logs/training_log.json',
            'description': 'Training progress log',
            'versioned': True  # Updates with each training run
        }
    }

    def __init__(self, bucket_name: str = None):
        self.bucket_name = bucket_name or os.getenv("S3_MODEL_BUCKET")

        if not self.bucket_name:
            raise ValueError(
                "S3_MODEL_BUCKET not set. Add to .env:\n"
                "S3_MODEL_BUCKET=your-team-bucket-name"
            )

        try:
            # Use AWS_PROFILE from .env if set
            aws_profile = os.getenv("AWS_PROFILE")
            if aws_profile:
                session = boto3.Session(profile_name=aws_profile)
                self.s3 = session.client('s3')
            else:
                self.s3 = boto3.client('s3')

            # Test connection
            self.s3.head_bucket(Bucket=self.bucket_name)
        except NoCredentialsError:
            raise ValueError(
                "AWS credentials not found. Run:\n"
                "  aws configure --profile hackathon\n"
                "Or set AWS_PROFILE=hackathon in .env"
            )
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise ValueError(f"Bucket {self.bucket_name} does not exist. Create it first.")
            raise

    def _compute_checksum(self, filepath: str) -> str:
        """Compute MD5 checksum of a file."""
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _get_metadata(self, asset_name: str) -> dict:
        """Get metadata for an asset from S3."""
        try:
            meta_key = f"metadata/{asset_name}_metadata.json"
            response = self.s3.get_object(Bucket=self.bucket_name, Key=meta_key)
            return json.loads(response['Body'].read())
        except ClientError:
            return None

    def _save_metadata(self, asset_name: str, metadata: dict):
        """Save metadata for an asset to S3."""
        meta_key = f"metadata/{asset_name}_metadata.json"
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=meta_key,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )

    def upload(self, asset_name: str, message: str = None, force: bool = False) -> bool:
        """Upload an asset to S3."""
        if asset_name not in self.ASSETS:
            print(f"Unknown asset: {asset_name}. Choose from: {list(self.ASSETS.keys())}")
            return False

        asset = self.ASSETS[asset_name]
        local_path = asset['local']
        s3_key = asset['s3_key']

        # Check if local exists
        if not os.path.exists(local_path):
            print(f"Local {asset['description']} not found at: {local_path}")
            return False

        # Handle directory (PDF store)
        if os.path.isdir(local_path):
            return self._upload_directory(asset_name, local_path, s3_key, message, force)

        # Handle single file
        return self._upload_file(asset_name, local_path, s3_key, message, force)

    def _upload_file(self, asset_name: str, local_path: str, s3_key: str, message: str, force: bool) -> bool:
        """Upload a single file."""
        # Check if remote is newer
        if not force:
            remote_meta = self._get_metadata(asset_name)
            if remote_meta:
                local_checksum = self._compute_checksum(local_path)
                if local_checksum == remote_meta.get('checksum'):
                    print(f"{self.ASSETS[asset_name]['description']} is already up-to-date in S3")
                    return True

                response = input(f"Remote version exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("Upload cancelled")
                    return False

        # Prepare metadata
        checksum = self._compute_checksum(local_path)
        file_size = os.path.getsize(local_path)

        metadata = {
            'asset': asset_name,
            'timestamp': datetime.now().isoformat(),
            'author': os.getenv('USER', 'unknown'),
            'checksum': checksum,
            'size_bytes': file_size,
            'message': message or f"Upload {asset_name}"
        }

        # Upload
        print(f"Uploading {self.ASSETS[asset_name]['description']}...")
        print(f"  {local_path} -> s3://{self.bucket_name}/{s3_key}")
        print(f"  Size: {file_size / (1024*1024):.2f} MB")

        try:
            self.s3.upload_file(
                local_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={'Metadata': {'checksum': checksum, 'author': metadata['author']}}
            )

            self._save_metadata(asset_name, metadata)

            print(f"Successfully uploaded {asset_name}")
            return True

        except ClientError as e:
            print(f"Error uploading: {e}")
            return False

    def _upload_directory(self, asset_name: str, local_dir: str, s3_prefix: str, message: str, force: bool) -> bool:
        """Upload a directory (for PDF store)."""
        if not s3_prefix.endswith('/'):
            s3_prefix += '/'

        files_to_upload = []
        total_size = 0

        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_dir)
                s3_key = s3_prefix + relative_path.replace('\\', '/')
                files_to_upload.append((local_file, s3_key))
                total_size += os.path.getsize(local_file)

        if not files_to_upload:
            print(f"No files found in {local_dir}")
            return False

        print(f"Uploading {len(files_to_upload)} files ({total_size / (1024*1024):.2f} MB)...")

        metadata = {
            'asset': asset_name,
            'timestamp': datetime.now().isoformat(),
            'author': os.getenv('USER', 'unknown'),
            'file_count': len(files_to_upload),
            'total_size_bytes': total_size,
            'message': message or f"Upload {asset_name}"
        }

        try:
            for local_file, s3_key in files_to_upload:
                print(f"  {os.path.basename(local_file)}...")
                self.s3.upload_file(local_file, self.bucket_name, s3_key)

            self._save_metadata(asset_name, metadata)

            print(f"Successfully uploaded {asset_name}")
            return True

        except ClientError as e:
            print(f"Error uploading: {e}")
            return False

    def download(self, asset_name: str, force: bool = False) -> bool:
        """Download an asset from S3."""
        if asset_name not in self.ASSETS:
            print(f"Unknown asset: {asset_name}. Choose from: {list(self.ASSETS.keys())}")
            return False

        asset = self.ASSETS[asset_name]
        local_path = asset['local']
        s3_key = asset['s3_key']

        # Check if remote exists
        remote_meta = self._get_metadata(asset_name)
        if not remote_meta:
            print(f"No remote {asset['description']} found in S3")
            return False

        # Check if already up-to-date
        if not force and os.path.exists(local_path):
            if os.path.isfile(local_path):
                local_checksum = self._compute_checksum(local_path)
                if local_checksum == remote_meta.get('checksum'):
                    print(f"{asset['description']} is already up-to-date")
                    return True

        # Handle directory download
        if s3_key.endswith('/'):
            return self._download_directory(asset_name, local_path, s3_key)

        # Handle single file download
        return self._download_file(asset_name, local_path, s3_key, remote_meta)

    def _download_file(self, asset_name: str, local_path: str, s3_key: str, remote_meta: dict) -> bool:
        """Download a single file."""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading {self.ASSETS[asset_name]['description']}...")
        print(f"  s3://{self.bucket_name}/{s3_key} -> {local_path}")
        print(f"  Size: {remote_meta['size_bytes'] / (1024*1024):.2f} MB")
        print(f"  Version: {remote_meta['timestamp']} by {remote_meta.get('author', 'unknown')}")

        try:
            self.s3.download_file(self.bucket_name, s3_key, local_path)

            # Verify checksum
            local_checksum = self._compute_checksum(local_path)
            if local_checksum != remote_meta['checksum']:
                print("Warning: Checksum mismatch!")
                return False

            print(f"Successfully downloaded {asset_name}")
            return True

        except ClientError as e:
            print(f"Error downloading: {e}")
            return False

    def _download_directory(self, asset_name: str, local_dir: str, s3_prefix: str) -> bool:
        """Download a directory (for PDF store)."""
        print(f"Downloading {self.ASSETS[asset_name]['description']}...")

        try:
            # List all objects with prefix
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)

            files_downloaded = 0
            for page in pages:
                for obj in page.get('Contents', []):
                    s3_key = obj['Key']
                    relative_path = s3_key[len(s3_prefix):]

                    if not relative_path:  # Skip the directory itself
                        continue

                    local_file = os.path.join(local_dir, relative_path)
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)

                    print(f"  {relative_path}...")
                    self.s3.download_file(self.bucket_name, s3_key, local_file)
                    files_downloaded += 1

            print(f"Successfully downloaded {files_downloaded} files")
            return True

        except ClientError as e:
            print(f"Error downloading: {e}")
            return False

    def status(self, asset_name: str = None):
        """Show sync status for asset(s)."""
        print("=" * 80)
        print("S3 SYNC STATUS")
        print("=" * 80)

        assets_to_check = [asset_name] if asset_name else self.ASSETS.keys()

        for name in assets_to_check:
            asset = self.ASSETS[name]
            print(f"\n{asset['description'].upper()}")
            print("-" * 80)

            # Local status
            local_path = asset['local']
            if os.path.exists(local_path):
                if os.path.isfile(local_path):
                    size = os.path.getsize(local_path)
                    checksum = self._compute_checksum(local_path)
                    print(f"Local:  EXISTS  {size / (1024*1024):.2f} MB  [{checksum[:12]}...]")
                else:
                    file_count = sum(len(files) for _, _, files in os.walk(local_path))
                    print(f"Local:  EXISTS  {file_count} files")
            else:
                print(f"Local:  NOT FOUND")

            # Remote status
            remote_meta = self._get_metadata(name)
            if remote_meta:
                size_mb = remote_meta.get('size_bytes', 0) / (1024*1024)
                timestamp = remote_meta.get('timestamp', 'unknown')
                author = remote_meta.get('author', 'unknown')
                checksum = remote_meta.get('checksum', 'N/A')

                if 'file_count' in remote_meta:
                    print(f"Remote: EXISTS  {remote_meta['file_count']} files  "
                          f"{timestamp[:10]} by {author}")
                else:
                    print(f"Remote: EXISTS  {size_mb:.2f} MB  [{checksum[:12]}...]  "
                          f"{timestamp[:10]} by {author}")
            else:
                print(f"Remote: NOT FOUND")

            # Recommendation
            local_exists = os.path.exists(local_path)
            remote_exists = remote_meta is not None

            if not local_exists and remote_exists:
                print(f"Action: DOWNLOAD (local missing)")
            elif local_exists and not remote_exists:
                print(f"Action: UPLOAD (remote missing)")
            elif local_exists and remote_exists:
                if os.path.isfile(local_path):
                    local_checksum = self._compute_checksum(local_path)
                    if local_checksum == remote_meta.get('checksum'):
                        print(f"Action: IN SYNC")
                    else:
                        print(f"Action: DIFFERS (check which is newer)")
                else:
                    print(f"Action: CHECK MANUALLY (directory)")
            else:
                print(f"Action: NEED TO CREATE")

        print("=" * 80)

    def sync_all_download(self):
        """Download all available assets from S3."""
        print("\nSyncing all assets from S3...\n")

        for name, asset in self.ASSETS.items():
            print(f"Checking {asset['description']}...")
            self.download(name, force=False)
            print()

    def sync_all_upload(self, message: str = None):
        """Upload all available assets to S3."""
        print("\nUploading all assets to S3...\n")

        for name, asset in self.ASSETS.items():
            if os.path.exists(asset['local']):
                print(f"Uploading {asset['description']}...")
                self.upload(name, message=message, force=False)
                print()
            else:
                print(f"Skipping {asset['description']} (not found locally)\n")


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Team S3 sync for all assets")
    parser.add_argument('action', help="Action: status, sync-all, download-{asset}, upload-{asset}")
    parser.add_argument('-m', '--message', help="Commit message for uploads")
    parser.add_argument('-f', '--force', action='store_true', help="Force action")
    parser.add_argument('--bucket', help="Override S3 bucket name")

    args = parser.parse_args()

    try:
        sync = S3TeamSync(bucket_name=args.bucket)

        if args.action == 'status':
            sync.status()

        elif args.action == 'sync-all':
            sync.sync_all_download()

        elif args.action.startswith('download-'):
            asset = args.action.replace('download-', '')
            sync.download(asset, force=args.force)

        elif args.action.startswith('upload-'):
            asset = args.action.replace('upload-', '')
            sync.upload(asset, message=args.message, force=args.force)

        else:
            print(f"Unknown action: {args.action}")
            print("Available: status, sync-all, download-{{asset}}, upload-{{asset}}")
            print(f"Assets: {list(S3TeamSync.ASSETS.keys())}")
            exit(1)

    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()