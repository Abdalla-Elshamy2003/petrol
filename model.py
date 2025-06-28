import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
import plotly.express as px
import streamlit as st
import datetime

# --------------------- كارت ملوّن جاهز للاستخدام ---------------------
def render_colored_card(title, value, subtitle, color1, color2, radius="2.2rem", icon=""):
    return f"""
    <div style="
        border-radius: {radius};
        background: linear-gradient(135deg, {color1} 0%, {color2} 100%);
        box-shadow: 0 4px 32px 0 rgba(50,50,93,.08), 0 1.5px 2.5px 0 rgba(0,0,0,.07);
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        margin-bottom: 1.8rem;
        text-align:center;
        position:relative;
    ">
        <div style='font-size:2.1rem; font-weight:800; color:#fff; letter-spacing:1.5px; margin-bottom:0.5rem;'>{icon} {title}</div>
        <div style='font-size:2.2rem; font-weight:700; color:#222; margin-bottom:0.7rem;'>{value}</div>
        <div style='font-size:1.1rem; color:#fff; letter-spacing:1.1px'>{subtitle}</div>
    </div>
    """

MODEL_DIR = "model_artifacts"
PREDICTIONS_DIR = "predictions"
for directory in [MODEL_DIR, PREDICTIONS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        st.error(f"خطأ في تحميل الملف: {e}")
        return None
    cols_to_drop = list(range(8, 28))
    df = df.drop(df.columns[cols_to_drop], axis=1)
    df = df.dropna(subset=['السنة', 'الشهر', 'رقم الحساب', 'اسم الزبون', 'اسم المنتج', 'وصف التعبئة'])
    for col in ['المنطقة', 'اسم الزبون', 'اسم المنتج', 'وصف التعبئة']:
        df[col] = df[col].astype(str).str.strip()
    unknown_indices = df[df['المنطقة'] == 'Unknown'].index
    n_unknown = len(unknown_indices)
    regions = [1.0, 2.0]
    assigned_regions = np.tile(regions, int(np.ceil(n_unknown / len(regions))))[:n_unknown]
    np.random.shuffle(assigned_regions)
    df.loc[unknown_indices, 'المنطقة'] = assigned_regions
    return df

def prepare_and_train_model(df):
    Q1 = df['الكمية باللتر'].quantile(0.25)
    Q3 = df['الكمية باللتر'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df[(df['الكمية باللتر'] >= lower_bound) & (df['الكمية باللتر'] <= upper_bound)].copy()
    df_clean['الربع'] = ((df_clean['الشهر'] - 1) // 3 + 1).astype(int)
    df_clean['موسم_الصيف'] = df_clean['الشهر'].apply(lambda x: 1 if x in [6, 7, 8, 9] else 0)
    df_clean['تكرار_العميل'] = df_clean.groupby('اسم الزبون')['اسم الزبون'].transform('count')
    df_clean['تكرار_المنتج'] = df_clean.groupby('اسم المنتج')['اسم المنتج'].transform('count')
    top_clients = df_clean['اسم الزبون'].value_counts().nlargest(10).index
    df_clean['العميل_Top'] = df_clean['اسم الزبون'].apply(lambda x: x if x in top_clients else 'Other')
    df_clean = df_clean.sort_values(['اسم الزبون', 'السنة', 'الشهر'])
    df_clean['كمية_سابقة'] = df_clean.groupby('اسم الزبون')['الكمية باللتر'].shift(1)
    df_clean['log_كمية_سابقة'] = np.log1p(df_clean['كمية_سابقة'])
    df_clean['log_كمية_سابقة'] = df_clean['log_كمية_سابقة'].fillna(df_clean['log_كمية_سابقة'].mean())
    label_cols = ['المنطقة', 'اسم المنتج', 'وصف التعبئة', 'العميل_Top']
    label_encoders = {}
    for col in label_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le
        joblib.dump(le, os.path.join(MODEL_DIR, f"{col}_encoder.pkl"))
    numeric_cols = ['السنة', 'الشهر', 'رقم الحساب', 'الربع', 'تكرار_العميل', 'تكرار_المنتج', 'log_كمية_سابقة']
    scaler = StandardScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_cols = ['المنطقة', 'اسم المنتج', 'وصف التعبئة', 'العميل_Top', 'السنة', 'الشهر',
                    'رقم الحساب', 'الربع', 'موسم_الصيف', 'تكرار_العميل', 'تكرار_المنتج', 'log_كمية_سابقة']
    X = df_clean[feature_cols]
    y = df_clean['الكمية باللتر']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42,
                        reg_alpha=1.0, reg_lambda=1.0)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return model, label_encoders, scaler, feature_cols, numeric_cols, top_clients, df_clean, rmse, r2

def retrieve_or_predict(row, orig_df, model, label_encoders, scaler, feature_cols, numeric_cols, top_clients, df_clean):
    row = row.copy()
    for col in ['المنطقة', 'اسم الزبون', 'اسم المنتج', 'وصف التعبئة']:
        row[col] = str(row[col]).strip()
    row['السنة'] = int(row['السنة'])
    row['الشهر'] = int(row['الشهر'])
    row['رقم الحساب'] = int(row['رقم الحساب'])
    lookup_cols = ['المنطقة', 'السنة', 'الشهر', 'رقم الحساب', 'اسم الزبون', 'اسم المنتج', 'وصف التعبئة']
    matched = orig_df
    for col in lookup_cols:
        matched = matched[matched[col] == row[col]]
    if not matched.empty:
        return float(matched.iloc[0]['الكمية باللتر']), "استرجاع"
    tmp = row.copy()
    tmp['الربع'] = ((tmp['الشهر'] - 1) // 3 + 1)
    tmp['موسم_الصيف'] = 1 if tmp['الشهر'] in [6, 7, 8, 9] else 0
    tmp['تكرار_العميل'] = df_clean['اسم الزبون'].value_counts().get(tmp['اسم الزبون'], 1)
    tmp['تكرار_المنتج'] = df_clean['اسم المنتج'].value_counts().get(tmp['اسم المنتج'], 1)
    tmp['العميل_Top'] = tmp['اسم الزبون'] if tmp['اسم الزبون'] in top_clients else 'Other'
    tmp['log_كمية_سابقة'] = df_clean['log_كمية_سابقة'].mean()
    for col in label_encoders:
        le = label_encoders[col]
        if tmp[col] not in le.classes_ and 'Other' in le.classes_:
            tmp[col] = 'Other'
        elif tmp[col] not in le.classes_:
            tmp[col] = le.classes_[0]
        tmp[col] = le.transform([str(tmp[col])])[0]
    for col in numeric_cols:
        tmp[col] = float(tmp[col])
    tmp_arr = np.array([tmp[col] for col in numeric_cols]).reshape(1, -1)
    tmp_scaled = scaler.transform(tmp_arr)[0]
    for i, col in enumerate(numeric_cols):
        tmp[col] = tmp_scaled[i]
    tmp_X = pd.DataFrame([tmp])[feature_cols]
    return float(model.predict(tmp_X)[0]), "توقع"

def save_prediction(data, prediction, source):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_df = pd.DataFrame([{
        **data,
        'الكمية_المتوقعة_باللتر': prediction,
        'مصدر': source,
        'تاريخ_التوقع': timestamp
    }])
    prediction_file = os.path.join(PREDICTIONS_DIR, f"predictions_{timestamp}.csv")
    prediction_df.to_csv(prediction_file, index=False, encoding='utf-8-sig')
    return prediction_file

# -------------------- واجهة Streamlit الرئيسية --------------------
def main():
    st.set_page_config(page_title="توقع وتحليل كمية الديزل", layout="wide")

    # عنوان جذاب
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #32B67A 10%, #72D7B2 90%);
                    border-radius:2.5rem;
                    padding: 1rem 2.2rem;
                    margin-bottom:1.6rem;
                    box-shadow:0 8px 24px 0 rgba(50,50,93,.09);
                    text-align:center;">
            <span style="font-size:2.6rem; font-weight:900; color:#fff; letter-spacing:2px;">
                🚚 توقع كمية الديزل الذكية
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    orig_df = load_and_preprocess_data("ديزل 20-24.xlsx")
    if orig_df is None:
        return
    model, label_encoders, scaler, feature_cols, numeric_cols, top_clients, df_clean, rmse, r2 = prepare_and_train_model(orig_df)

    page = st.sidebar.selectbox("اختر الصفحة", ["تحليل البيانات", "توقع الكمية"])

    if page == "تحليل البيانات":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(render_colored_card(
                "إجمالي الكمية", f"{int(df_clean['الكمية باللتر'].sum()):,}", "إجمالي الديزل الموزع",
                "#2196F3", "#1565C0", icon="🛢️"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_colored_card(
                "عدد العملاء", df_clean["اسم الزبون"].nunique(), "عملاء فريدون",
                "#43A047", "#166534", icon="👤"), unsafe_allow_html=True)
        with col3:
            # أكثر منتج
            top_product_row = df_clean.groupby('اسم المنتج')['الكمية باللتر'].sum().idxmax()
            top_product_val = int(df_clean.groupby('اسم المنتج')['الكمية باللتر'].sum().max())
            st.markdown(render_colored_card(
                "الأكثر مبيعًا", f"{top_product_row}", f"{top_product_val:,} لتر",
                "#FFC107", "#FF6F00", icon="🏆"), unsafe_allow_html=True)
        with col4:
            # أكثر عميل
            top_customer_row = df_clean.groupby('اسم الزبون')['الكمية باللتر'].sum().idxmax()
            top_customer_val = int(df_clean.groupby('اسم الزبون')['الكمية باللتر'].sum().max())
            st.markdown(render_colored_card(
                "أكبر عميل", f"{top_customer_row}", f"{top_customer_val:,} لتر",
                "#EC407A", "#7B1FA2", icon="🥇"), unsafe_allow_html=True)

        # Pie chart and histogram
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            region_counts = df_clean['المنطقة'].value_counts()
            fig1 = px.pie(names=region_counts.index, values=region_counts.values, title='توزيع المناطق',
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig1.update_traces(textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
        with col_viz2:
            fig2 = px.histogram(df_clean, x='الكمية باللتر', title='توزيع الكمية باللتر',
                               nbins=20, color_discrete_sequence=['#FF6F61'])
            fig2.update_layout(xaxis_title='الكمية باللتر', yaxis_title='عدد الصفوف')
            st.plotly_chart(fig2, use_container_width=True)
        # Monthly trend
        monthly_avg = df_clean.groupby('الشهر')['الكمية باللتر'].mean().reset_index()
        fig3 = px.line(monthly_avg, x='الشهر', y='الكمية باللتر', title='متوسط الكمية حسب الشهر',
                      line_shape='spline', color_discrete_sequence=['#43A047'])
        st.plotly_chart(fig3, use_container_width=True)
        # Top customers
        top_customers = df_clean.groupby('اسم الزبون')['الكمية باللتر'].sum().nlargest(10).reset_index()
        fig4 = px.bar(top_customers, x='اسم الزبون', y='الكمية باللتر', title='أعلى 10 عملاء حسب الكمية',
                     color_discrete_sequence=['#FFB300'])
        fig4.update_layout(xaxis_title='اسم الزبون', yaxis_title='الكمية باللتر', xaxis_tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)
        # Yearly trend
        yearly_sum = df_clean.groupby('السنة')['الكمية باللتر'].sum().reset_index()
        fig5 = px.line(yearly_sum, x='السنة', y='الكمية باللتر', title='إجمالي الكمية حسب السنة',
                      line_shape='spline', color_discrete_sequence=['#8E24AA'])
        st.plotly_chart(fig5, use_container_width=True)

    else:
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                region = st.selectbox("المنطقة", options=[1.0, 2.0])
                year = st.number_input("السنة", min_value=2020, max_value=2025, step=1, value=2023)
                month = st.number_input("الشهر", min_value=1, max_value=12, step=1, value=1)
            with col2:
                account = st.number_input("رقم الحساب", min_value=0, step=1, value=170003)
                customer = st.text_input("اسم الزبون", value="محطة كهرباءغرب طرابلس")
                product = st.text_input("اسم المنتج", value="وقود الديزل")
                packaging = st.text_input("وصف التعبئة", value="سائب/لتر")
            submitted = st.form_submit_button("توقع الكمية")
        if submitted:
            if not customer or not product or not packaging:
                st.error("يرجى ملء جميع الحقول")
            else:
                new_data = pd.DataFrame([{
                    'المنطقة': region,
                    'السنة': year,
                    'الشهر': month,
                    'رقم الحساب': account,
                    'اسم الزبون': customer,
                    'اسم المنتج': product,
                    'وصف التعبئة': packaging
                }])
                prediction, source = retrieve_or_predict(
                    new_data.iloc[0], orig_df, model, label_encoders, scaler,
                    feature_cols, numeric_cols, top_clients, df_clean
                )
                prediction_file = save_prediction(new_data.iloc[0], prediction, source)
                st.session_state.prediction_history.append({
                    **new_data.iloc[0],
                    'الكمية_المتوقعة_باللتر': prediction,
                    'مصدر': source
                })
                st.markdown(render_colored_card(
                    "🔥 النتيجة المتوقعة", f"{prediction:,.2f} لتر", f"المصدر: {source}",
                    "#57C1EB", "#246FA8", radius="2.5rem", icon="⛽️"
                ), unsafe_allow_html=True)
                with open(prediction_file, 'rb') as f:
                    st.download_button(
                        label="تحميل التوقع",
                        data=f,
                        file_name=os.path.basename(prediction_file),
                        mime="text/csv"
                    )
        if st.session_state.prediction_history:
            st.markdown(
                '<div style="font-size:1.6rem; font-weight:700; color:#1565C0; margin-bottom:1rem;">سجل التوقعات</div>',
                unsafe_allow_html=True
            )
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)

if __name__ == "__main__":
    main()
