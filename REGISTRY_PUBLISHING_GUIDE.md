# 📝 ComfyUI Registry Publishing Guide

**Official guide for publishing to the ComfyUI Registry**

---

## ✅ Prerequisites Completed

- ✅ comfy-cli installed
- ✅ pyproject.toml created and configured
- ✅ Repository on GitHub
- ✅ All code and documentation ready

---

## 🚀 Step-by-Step Publishing Process

### Step 1: Create Publisher Account

1. **Go to:** https://registry.comfy.org/

2. **Sign in** with GitHub

3. **Create Publisher Profile:**
   - Click on your profile/settings
   - Create a publisher identity
   - **Important:** Your Publisher ID appears after the `@` symbol
   - Example: `@jonstreeter`
   - **This ID is permanent and cannot be changed!**

4. **Copy your Publisher ID** (without the @)

---

### Step 2: Update pyproject.toml with Your Publisher ID

1. Open: `pyproject.toml`

2. Find the line:
   ```toml
   PublisherId = "jonstreeter"  # IMPORTANT: Replace with your actual Publisher ID
   ```

3. **Replace `"jonstreeter"`** with your actual Publisher ID from the registry

4. Save the file

---

### Step 3: Generate API Key

1. **Go to:** https://registry.comfy.org/ (your publisher settings)

2. **Find:** "Registry Publishing API Key" section

3. **Click:** "Generate API Key" or "Create New Key"

4. **IMPORTANT:** Copy this key immediately!
   - You cannot view it again
   - If lost, you must generate a new one
   - Store it securely (password manager recommended)

5. **This key is for:**
   - ✅ Publishing custom nodes to Registry
   - ✅ Publishing to ComfyUI-Manager
   - ❌ NOT for paid API node usage (different key)

---

### Step 4: Publish Your Node

**Option A: Manual Publishing (First Time)**

```bash
# Navigate to your node directory
cd "G:\AIART\AI_Image_Generators\Comfy_UI_V42\ComfyUI\custom_nodes\ComfyUI-Deep-Exemplar-based-Video-Colorization"

# Publish to registry
comfy node publish

# When prompted, paste your API key
```

**Option B: Automated GitHub Actions (Recommended)**

See "GitHub Actions Setup" section below.

---

### Step 5: Verify Publication

After publishing, your node will appear at:
```
https://registry.comfy.org/@YOUR_PUBLISHER_ID/comfyui-reference-video-colorization
```

---

## 🤖 GitHub Actions Setup (Automated Publishing)

### Benefits:
- ✅ Automatic publishing on new releases
- ✅ No manual steps needed
- ✅ Consistent versioning
- ✅ Triggered by pyproject.toml changes

### Setup:

1. **Add API Key to GitHub Secrets:**
   - Go to: https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization/settings/secrets/actions
   - Click: "New repository secret"
   - Name: `REGISTRY_ACCESS_TOKEN`
   - Value: [Your API key from Step 3]
   - Click: "Add secret"

2. **Create Workflow File:**

Create: `.github/workflows/publish-to-registry.yml`

```yaml
name: Publish to ComfyUI Registry

on:
  push:
    branches:
      - main
    paths:
      - 'pyproject.toml'
  workflow_dispatch:

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install comfy-cli
        run: pip install comfy-cli

      - name: Publish to registry
        env:
          REGISTRY_ACCESS_TOKEN: ${{ secrets.REGISTRY_ACCESS_TOKEN }}
        run: comfy node publish
```

3. **Commit and Push:**
```bash
git add .github/workflows/publish-to-registry.yml pyproject.toml
git commit -m "Add automated registry publishing"
git push origin main
```

---

## 📊 Current Configuration

**Node Name:** `comfyui-reference-video-colorization`

**Display Name:** `Reference-Based Video Colorization`

**Version:** `2.0.0`

**Repository:** https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization

**Icon:** `Workflows/Reference-Based-Colorization-Workflow.png`

**Description:**
```
Dual implementation of reference-based video colorization featuring
ColorMNet (2024) with DINOv2 and Deep Exemplar (2019). Includes 4 nodes,
multiple feature encoders, advanced post-processing, and auto-installer.
```

**Dependencies:**
- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.24.0
- Pillow>=9.5.0
- scikit-image>=0.21.0
- opencv-python>=4.8.0
- einops>=0.7.0
- tqdm>=4.65.0
- progressbar2>=4.2.0
- gdown

---

## 🔄 Updating Your Published Node

### When you make changes:

1. **Update version in pyproject.toml:**
   ```toml
   version = "2.0.1"  # Increment version
   ```

2. **Commit changes:**
   ```bash
   git add .
   git commit -m "Update to v2.0.1: [describe changes]"
   git push origin main
   ```

3. **Publish update:**
   - **Manual:** Run `comfy node publish` again
   - **Automated:** GitHub Actions will publish automatically

### Version Guidelines:
- **Major (2.0.0 → 3.0.0):** Breaking changes
- **Minor (2.0.0 → 2.1.0):** New features, backward compatible
- **Patch (2.0.0 → 2.0.1):** Bug fixes, backward compatible

---

## ✅ Pre-Publication Checklist

Before running `comfy node publish`:

- [x] Publisher account created on registry.comfy.org
- [ ] Publisher ID updated in pyproject.toml
- [ ] API key generated and saved securely
- [x] pyproject.toml properly configured
- [x] LICENSE file exists
- [x] README.md is comprehensive
- [x] All code tested and working
- [x] Example workflows included
- [x] Dependencies documented
- [x] Repository is public

---

## 🎯 Quick Start Commands

```bash
# 1. Navigate to node directory
cd "G:\AIART\AI_Image_Generators\Comfy_UI_V42\ComfyUI\custom_nodes\ComfyUI-Deep-Exemplar-based-Video-Colorization"

# 2. Verify pyproject.toml
cat pyproject.toml

# 3. Update Publisher ID (if needed)
# Edit pyproject.toml and replace "jonstreeter" with your actual Publisher ID

# 4. Publish
comfy node publish
# (Paste your API key when prompted)

# 5. Verify
# Visit: https://registry.comfy.org/@YOUR_PUBLISHER_ID/comfyui-reference-video-colorization
```

---

## 🔗 Important Links

- **ComfyUI Registry:** https://registry.comfy.org/
- **Publishing Docs:** https://docs.comfy.org/registry/publishing
- **Your Repository:** https://github.com/jonstreeter/ComfyUI-Reference-Based-Video-Colorization
- **Your Publisher Profile:** https://registry.comfy.org/@YOUR_PUBLISHER_ID

---

## ❓ Troubleshooting

### "Publisher ID not found"
- Make sure you've created a publisher account at registry.comfy.org
- Verify the Publisher ID in pyproject.toml matches your registry profile
- Publisher ID is case-sensitive

### "Invalid API key"
- Regenerate API key from registry.comfy.org
- Make sure you're using the Registry Publishing API key (not paid API key)
- Check for extra spaces when copying

### "pyproject.toml validation failed"
- Run `comfy node init` to see example format
- Check that all required fields are present
- Ensure version follows semantic versioning (X.Y.Z)

### "Icon not found"
- Make sure the icon path is relative to the repository root
- Supported formats: SVG, PNG, JPG, GIF
- Max size: 800x400px

---

## 🎉 Success!

Once published, users can install your node via:

1. **ComfyUI Manager** (will be automatically indexed)
2. **comfy-cli:**
   ```bash
   comfy node install comfyui-reference-video-colorization
   ```
3. **Manual git clone** (existing method still works)

---

**Your node will appear in the official ComfyUI Registry!** 🚀
