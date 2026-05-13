# User manual — Image affinity ranker

This app scores and ranks up to five images against a customer profile (numerical and categorical fields), then shows results on a separate page. You need **at least one image** and a **valid profile** (from CSV or from the form) before you run **Predict**.

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

Open the **Customer profile** tab. You can define the profile in **one** of two ways. If you attach a CSV file, it **overrides** whatever is entered in the form.

Fields come from `app/user_features_manifest.json` and are compiled into `app/profile_attributes.json`:

- **Information (`inf__…`)** — demographics-style inputs at the top of the tab.
- **Preference (`pref__…`)** — taste / media preference inputs below.

### 3a. Use the form (default)

1. **Numerical** rows show a number box (for example age, hours, income). Values must be **zero or positive** (non-negative); the browser also sets `min="0"`, and the server rejects negatives. Defaults start at **0**.
2. **Categorical** rows show a **dropdown**. Each choice has a **stored value** (the full multihot token, e.g. `inf__fave_sports__individual_sports_tennis_archery`) and a **display label** defined in `profile_attributes.json` under `options` as `{"value": "...", "label": "..."}`. Edit `label` to any text you want without changing `value` (CSV and form submissions still use `value`).
3. **Default for dropdowns:** the **first** option in the list for that attribute (manifest order within the group) is pre-selected.

### 3b. Upload a profile CSV (optional)

Use this when you already have a profile row in a spreadsheet or export.

**Requirements:**

- File must be plain text **UTF-8** CSV.
- **First row:** column headers only. Header names and **order** must match the app exactly: the `id` values in `app/profile_attributes.json` (one column per numerical field and one column per **grouped** categorical attribute such as `inf__fave_sports`, `pref__most_read_books`, etc.).
- **Second row:** exactly **one** data row. Numerical cells must be **finite and non-negative**. Categorical cells must be one of the allowed **value** strings for that column (the `value` field from `options`, not the display `label`).

**Example file in this repository:** `app/sample_valid_profile.csv` — you can upload that file as-is to satisfy the validator, or copy its structure for your own data.

You can also download a fresh template from the running app: open the link **“this template”** on the Customer profile tab, or go to `/profile/csv-template` in the browser.

**Regenerating profile metadata after manifest changes:** from the repository root, run  
`python3 app/scripts/build_profile_attributes.py`  
to rebuild `profile_attributes.json` and `sample_valid_profile.csv` from `user_features_manifest.json`.

**Important:** If you select a CSV under **Profile CSV (optional)**, the server **ignores** the form fields and uses only the CSV row (after validating it).

---

## 4. Run prediction

1. Confirm the **Images** tab has at least one file attached.
2. Confirm the **Customer profile** tab: numbers and dropdowns look correct, or you have attached a **valid CSV**.
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
| Profile via form | Non-negative number inputs; categorical dropdowns use explicit `value` / `label` pairs from JSON. |
| Profile via CSV | Overrides the form; see `app/sample_valid_profile.csv`. |
| Predict | Submits everything to the server and opens ranked results. |

For developers: rebuild `app/profile_attributes.json` from `app/user_features_manifest.json` using `app/scripts/build_profile_attributes.py` whenever the manifest changes.
