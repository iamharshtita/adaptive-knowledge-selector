# Quick Visualization Guide

## ✅ **FIXED VERSION - Use This!**

The blank screen issue has been fixed. Use this command:

```bash
cd "/Users/harshtita/Desktop/ASU Docs/KRR - Spring26/Course Project"

# Create visualization (works properly now!)
python3 scripts/create_viz.py <drug_name>
```

---

## 🚀 **Quick Commands**

```bash
# Create visualizations
python3 scripts/create_viz.py metformin
python3 scripts/create_viz.py atorvastatin
python3 scripts/create_viz.py ibuprofen
python3 scripts/create_viz.py warfarin

# Open in browser
open visualizations/metformin.html
open visualizations/atorvastatin.html
```

---

## 📂 **Available Visualizations**

Already created and working:
- `visualizations/metformin.html`
- `visualizations/atorvastatin.html`
- `visualizations/ibuprofen.html`

---

## 🎯 **How to Use**

### **Step 1: Create Visualization**
```bash
python3 scripts/create_viz.py metformin
```

### **Step 2: Open in Browser**

**Method 1 - Command line:**
```bash
open visualizations/metformin.html
```

**Method 2 - Manual:**
1. Open your browser (Chrome, Safari, Firefox)
2. Press `Cmd+O` (or File > Open File)
3. Navigate to:
   ```
   /Users/harshtita/Desktop/ASU Docs/KRR - Spring26/Course Project/visualizations/
   ```
4. Select `metformin.html` and open

**Method 3 - Finder:**
1. Open Finder
2. Navigate to the `visualizations` folder
3. Double-click `metformin.html`

---

## 🎨 **What You'll See**

### **Interactive Features:**

1. **Nodes (Circles)**
   - 🔴 Red = Drug (center, larger)
   - 🟦 Teal = Diseases
   - 🟩 Green = Genes
   - 🟨 Yellow = Side Effects
   - Other colors = Other entity types

2. **Interactions:**
   - **Drag nodes** - Click and drag to move them around
   - **Click nodes** - See detailed information on the right
   - **Hover nodes** - They grow slightly
   - **Physics simulation** - Nodes push/pull each other

3. **Information Panel (Right Side)**
   - Shows selected node details
   - Lists relationships
   - Connection count

4. **Legend (Bottom Right)**
   - Color guide for node types

---

## 🔍 **Example: Metformin**

When you open `metformin.html`, you'll see:

- **Center**: Metformin (large red circle)
- **Around it**: 20-30 connected entities
  - Yellow circles = Side effects it causes
  - Green circles = Genes it affects
  - Teal circles = Diseases it treats

**Try this:**
1. Click on Metformin (center) → See all its relationships
2. Drag Metformin → Watch other nodes follow
3. Click on a side effect → See what drugs cause it
4. Drag nodes to different positions → Rearrange the layout

---

## 📊 **Available Drugs**

### **Common Drugs You Can Visualize:**

```bash
# Cardiovascular
python3 scripts/create_viz.py atorvastatin
python3 scripts/create_viz.py simvastatin
python3 scripts/create_viz.py amlodipine
python3 scripts/create_viz.py metoprolol
python3 scripts/create_viz.py warfarin

# Pain/Anti-inflammatory
python3 scripts/create_viz.py ibuprofen
python3 scripts/create_viz.py acetaminophen
python3 scripts/create_viz.py "acetylsalicylic acid"  # Aspirin

# Diabetes
python3 scripts/create_viz.py metformin
python3 scripts/create_viz.py glipizide
python3 scripts/create_viz.py pioglitazone

# Antibiotics
python3 scripts/create_viz.py amoxicillin
python3 scripts/create_viz.py ciprofloxacin
python3 scripts/create_viz.py azithromycin
```

---

## 🛠️ **Troubleshooting**

### **"Blank screen in browser"**
✅ **FIXED!** Use `create_viz.py` instead of `simple_html_viz.py`

### **"Drug not found"**
Search for it first:
```bash
python3 scripts/list_drugs.py --search <drug_name>
```

### **"File won't open"**
Try:
```bash
open -a "Google Chrome" visualizations/metformin.html
# or
open -a Safari visualizations/metformin.html
```

---

## 📖 **All Visualization Tools**

| Tool | Purpose | Command |
|------|---------|---------|
| **create_viz.py** | Interactive HTML (BEST!) | `python3 scripts/create_viz.py <drug>` |
| **visualize_graph.py** | Static PNG image | `python3 scripts/visualize_graph.py drug <drug>` |
| **view_knowledge_graph.py** | Text explorer | `python3 scripts/view_knowledge_graph.py interactive` |
| **list_drugs.py** | Find drug names | `python3 scripts/list_drugs.py --search <term>` |

---

## 🎓 **Next Steps**

1. ✅ **Open metformin.html** - See the working visualization
2. ✅ **Create more drugs** - Try different medications
3. ✅ **Explore relationships** - Click nodes to learn
4. ✅ **Compare drugs** - Open multiple tabs
5. 📝 **Use for project** - Understand graph structure for selector model

---

## 💡 **Pro Tips**

**1. Compare Similar Drugs:**
```bash
# Visualize all statins
python3 scripts/create_viz.py atorvastatin
python3 scripts/create_viz.py simvastatin
python3 scripts/create_viz.py pravastatin

# Open all three → Compare their side effects
```

**2. Find Related Drugs:**
```bash
# Search by drug class
python3 scripts/list_drugs.py --search statin

# Visualize each result
```

**3. Explore Relationships:**
```bash
# Create visualization
python3 scripts/create_viz.py metformin

# Open it
open visualizations/metformin.html

# Then:
# - Click Metformin → See all relationships
# - Click a disease it treats → See what else treats it
# - Click a side effect → See what else causes it
```

---

## ✨ **Summary**

**Working command:**
```bash
python3 scripts/create_viz.py metformin
open visualizations/metformin.html
```

**That's it!** The graph will load with interactive nodes you can drag and click.

---

**Your visualization is ready!** 🎉

Open `visualizations/metformin.html` in your browser now!
