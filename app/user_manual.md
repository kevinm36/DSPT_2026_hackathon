# User manual — Image affinity ranker

This app scores and ranks up to five images against a categorical customer profile, then shows results on a separate page. You need **at least one image** and a **valid profile** (from CSV or from the form) before you run **Predict**.

---

## 1. Open the app

With the server running, open the home page in your browser (for example `http://127.0.0.1:8000/`).

The screen has two tabs: **Images** and **Customer profile**. Use both before clicking **Predict**.

---

## 2. Upload images (Images tab)

1. Click the **Images** tab.
2. For each slot (**Image 1** through **Image 5**), use **Browse** / **Choose File** and pick a file from your computer.
3. You may attach **one to five** images. Leave unused slots empty.
4. Allowed types: **JPEG**, **PNG**, **WebP**, or **GIF**. Each file must be within the size limit enforced by the server (see app configuration if you hit an error).

**Tip:** Higher-ranked images in the results are those the model scores more strongly for the profile you supplied.

---

## 3. Set the customer profile (Customer profile tab)

Open the **Customer profile** tab. You can define the profile in **one** of two ways. If you attach a CSV file, it **overrides** whatever is selected in the dropdowns.

### 3a. Use the dropdowns (default)

1. Each row is one categorical attribute (label text comes from configuration).
2. Open each **dropdown** and pick the value that matches your customer.
3. **Default behavior:** When the page loads, every attribute is preset to the **first allowed value** in the list for that attribute (the first entry in `profile_attributes.json` for that field). Change any dropdown as needed before submitting.

### 3b. Upload a profile CSV (optional)

Use this when you already have a profile row in a spreadsheet or export.

**Requirements:**

- File must be plain text **UTF-8** CSV.
- **First row:** column headers only. Header names and order must match the app exactly (same as `profile_attributes.json` attribute ids: `attribute_1`, `attribute_2`, … through `attribute_10` with the current sample configuration).
- **Second row:** exactly **one** data row with one cell per column. Each value must be one of the allowed options for that attribute (no extra spaces unless they are part of the option text).

**Example file in this repository:** `app/sample_valid_profile.csv` — you can upload that file as-is to satisfy the validator, or copy its structure for your own data.

You can also download a fresh template from the running app: open the link **“this template”** on the Customer profile tab, or go to `/profile/csv-template` in the browser. That download includes the correct header row and a sample second row.

**Important:** If you select a CSV under **Profile CSV (optional)**, the server **ignores** the dropdown choices and uses only the CSV row (after validating it).

---

## 4. Run prediction

1. Confirm the **Images** tab has at least one file attached.
2. Confirm the **Customer profile** tab: either you rely on the **defaults / dropdowns**, or you have attached a **valid CSV**.
3. Click **Predict**.

On success, the browser moves to the **results** page: thumbnails are ordered by score (best first). Click a thumbnail to load **image-level attributes** and **prediction reasoning** for that image below the gallery.

If something is wrong (missing image, bad CSV header, invalid option text, wrong file type), you will see an error page with a short message and a link **Back to upload**.

---

## 5. Start over

From the results page, use **← New prediction** to return to the home page and submit a new batch.

---

## Quick reference

| Item | Notes |
|------|--------|
| Images field name | Multiple inputs named `images` (up to five files). |
| Profile via form | One `<select>` per attribute; defaults = first option each. |
| Profile via CSV | Overrides dropdowns; see `app/sample_valid_profile.csv`. |
| Predict | Submits everything to the server and opens ranked results. |

For developers: attribute names, order, and allowed values come from `app/profile_attributes.json`. If that file changes, CSV headers and dropdown options change with it; update any sample CSVs you keep for users accordingly.
