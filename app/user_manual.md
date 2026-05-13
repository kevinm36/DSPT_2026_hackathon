# User manual — Image affinity ranker

This app scores and ranks up to five images against a customer profile (numerical and categorical fields), then shows results on a separate page. You need **at least one image** and a **valid profile** in the form before you run **Predict**. (You can load a CSV locally in the browser to fill the form faster; **Predict** always submits whatever values are currently in the form.)

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

Open the **Customer profile** tab. Use the form fields directly, or use an optional **local-only** CSV file in your browser to populate those fields from a spreadsheet row (see below). You can edit any field after loading a CSV; **Predict** always uses the **current form values**.

Fields come from `app/user_features_manifest.json` and are compiled into `app/profile_attributes.json`:

- **Information (`inf__…`)** — demographics-style inputs at the top of the tab.
- **Preference (`pref__…`)** — taste / media preference inputs below.

### 3a. Use the form (default)

1. **Numerical** rows show a number box (for example age, hours, income). Values must be **zero or positive** (non-negative); the browser also sets `min="0"`, and the server rejects negatives. Defaults start at **0**.
2. **Categorical** rows show a **dropdown**. Each choice has a **stored value** (the full multihot token, e.g. `inf__fave_sports__individual_sports_tennis_archery`) and a **display label** defined in `profile_attributes.json` under `options` as `{"value": "...", "label": "..."}`. Edit `label` to any text you want without changing `value` (CSV and form submissions still use `value`).
3. **Default for dropdowns:** the **first** option in the list for that attribute (manifest order within the group) is pre-selected.

### 3b. Load a profile CSV in the browser (optional)

Use this when you already have a profile row in a spreadsheet or export and want to paste it into the form quickly. The CSV file is read **only in your browser** to fill the fields; it is **not** sent to the server on **Predict**.

**Requirements (for the file you pick locally):**

- File must be plain text **UTF-8** CSV.
- **First row:** column headers only. Header names and **order** must match the app exactly: the `id` values in `app/profile_attributes.json` (one column per numerical field and one column per **grouped** categorical attribute such as `inf__fave_sports`, `pref__most_read_books`, etc.).
- **Second row:** exactly **one** data row. Numerical cells should be **finite and non-negative** (invalid cells appear blank in the form until you fix them). Categorical cells should be one of the allowed **value** strings for that column (the `value` field from `options`, not the display `label`); unknown values map to a blank-looking **invalid** choice that you must correct before **Predict**.

**Example file in this repository:** `app/sample_valid_profile.csv` — open it in a spreadsheet or use it as a structural reference for your own CSV.

You can also download a fresh template from the running app: open the link **“this template”** on the Customer profile tab, or go to `/profile/csv-template` in the browser.

**Regenerating profile metadata after manifest changes:** from the repository root, run  
`python3 app/scripts/build_profile_attributes.py`  
to rebuild `profile_attributes.json` and `sample_valid_profile.csv` from `user_features_manifest.json`.

**Important:** After loading a CSV, review every field. **Predict** validates and uses **only** the values shown in the form. If any field is still invalid or blank, you will see an error listing those fields by name.

---

## 4. Run prediction

1. Confirm the **Images** tab has at least one file attached.
2. Confirm the **Customer profile** tab: numbers and dropdowns look correct (including anything you fixed after loading a CSV).
3. Click **Predict**.

On success, the browser moves to the **results** page: thumbnails are ordered by score (best first). Click a thumbnail to load **image-level attributes** and **prediction reasoning** for that image below the gallery.

If something is wrong (missing image, invalid or blank profile fields, wrong image file type, etc.), you will see an error page with a short message and a link **Back to upload**.

---

## 5. Start over

From the results page, use **← New prediction** to return to the home page and submit a new batch.

---

## Quick reference

| Item | Notes |
|------|--------|
| Images field name | Multiple inputs named `images` (up to five files). |
| Profile via form | Non-negative number inputs; categorical dropdowns use explicit `value` / `label` pairs from JSON. |
| Profile via CSV | **Local only:** fills the form from the first data row; edit as needed; see `app/sample_valid_profile.csv`. |
| Predict | Submits images + **current form field values**; opens ranked results. |

For developers: rebuild `app/profile_attributes.json` from `app/user_features_manifest.json` using `app/scripts/build_profile_attributes.py` whenever the manifest changes.
