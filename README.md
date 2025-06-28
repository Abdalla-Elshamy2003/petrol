
---

## Data Requirements

- The data Excel file must contain at least these columns (in Arabic):
    - `المنطقة` (Region)
    - `السنة` (Year)
    - `الشهر` (Month)
    - `رقم الحساب` (Account Number)
    - `اسم الزبون` (Customer Name)
    - `اسم المنتج` (Product Name)
    - `وصف التعبئة` (Packaging Description)
    - `الكمية باللتر` (Quantity in Liters)

- The app automatically cleans extra columns and handles missing/unknown values.

---

## Customization

- **UI Colors, Cards, and Layout** can be easily edited in the code for branding.
- **Model** (XGBoost) can be retrained with different parameters or replaced.
- **Data Visualizations** (Pie, Bar, Line, Histogram) can be changed or extended.

---

## Example Screenshots

> _(Add screenshots here to showcase the dashboard, prediction form, and charts!)_

---

## Troubleshooting

- If the app does **not start**:  
  Make sure all dependencies are installed and you’re running with Python 3.8–3.11.

- **Encoding Issues:**  
  Ensure your Excel file is UTF-8 encoded and all required columns exist.

- **FileNotFoundError:**  
  Make sure your data file (`ديزل 20-24.xlsx`) is in the same directory as your script.

---

 

## License

This project is open-source. You can use, modify, and share it freely.

