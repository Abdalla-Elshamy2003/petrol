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

from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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

    if df.shape[1] > 8:
        cols_to_drop = list(range(8, df.shape[1]))
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

def prepare_and_train_models(df):
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
    model_xgb = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, reg_alpha=1.0, reg_lambda=1.0)
    model_xgb.fit(X_train, y_train)
    joblib.dump(model_xgb, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    y_pred = model_xgb.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    arima_series = df.groupby(['السنة', 'الشهر'])['الكمية باللتر'].sum().reset_index()
    arima_series['ds'] = pd.to_datetime(dict(year=arima_series['السنة'], month=arima_series['الشهر'], day=1))
    arima_series = arima_series.set_index('ds')
    arima_y = arima_series['الكمية باللتر']
    arima_order = (1, 1, 1)
    arima_model = ARIMA(arima_y, order=arima_order)
    arima_result = arima_model.fit()
    joblib.dump(arima_result, os.path.join(MODEL_DIR, "arima_model.pkl"))

    X_ann = X.values
    y_ann = y.values
    ann_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_ann.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    ann_model.compile(optimizer='adam', loss='mse')
    ann_model.fit(X_ann, y_ann, epochs=40, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=5)])
    ann_model.save(os.path.join(MODEL_DIR, "ann_model.h5"))

    return {
        "xgboost": (model_xgb, label_encoders, scaler, feature_cols, numeric_cols, top_clients, df_clean, rmse, r2),
        "arima": (arima_result, arima_y),
        "ann": (ann_model, label_encoders, scaler, feature_cols, numeric_cols, top_clients, df_clean)
    }

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

def predict_arima(row, arima_model, arima_y):
    target_date = pd.Timestamp(year=int(row['السنة']), month=int(row['الشهر']), day=1)
    last_date = arima_y.index[-1]
    n_periods = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
    if n_periods <= 0:
        try:
            return float(arima_y.loc[target_date]), "استرجاع"
        except:
            return float(arima_y.mean()), "توقع متوسط"
    pred = arima_model.forecast(steps=n_periods)
    return float(pred.values[-1]), "توقع"

def predict_ann(row, model, label_encoders, scaler, feature_cols, numeric_cols, top_clients, df_clean):
    row = row.copy()
    for col in ['المنطقة', 'اسم الزبون', 'اسم المنتج', 'وصف التعبئة']:
        row[col] = str(row[col]).strip()
    row['السنة'] = int(row['السنة'])
    row['الشهر'] = int(row['الشهر'])
    row['رقم الحساب'] = int(row['رقم الحساب'])
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
    y_pred = model.predict(tmp_X.values)
    return float(y_pred[0][0]), "توقع"

def save_prediction(data, prediction, source, model_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_df = pd.DataFrame([{
        **data,
        'الكمية_المتوقعة_باللتر': prediction,
        'مصدر': source,
        'الموديل': model_name,
        'تاريخ_التوقع': timestamp
    }])
    prediction_file = os.path.join(PREDICTIONS_DIR, f"predictions_{timestamp}.csv")
    prediction_df.to_csv(prediction_file, index=False, encoding='utf-8-sig')
    return prediction_file

def insight_explanation(row, model_name, models, orig_df):
    tips = ""
    why = ""
    if model_name == "XGBoost":
        tips = (
            "- يفضل تعبئة كل البيانات خاصة (اسم العميل - المنتج - وصف التعبئة).\n"
            "- العميل لو من ضمن التوب 10 احتمال التوقع يكون أدق.\n"
            "- عوامل مثل الربع، موسم الصيف، وكمية العميل سابقًا تؤثر بقوة على النتيجة.\n"
            "- التوقع يعتمد على علاقات معقدة بين كل المدخلات بناء على تعلم النموذج."
        )
        why = (
            f"القيمة المتوقعة تعتمد بشكل رئيسي على :\n"
            f"- اسم العميل (تم اعتباره {'عميل رئيسي' if row['اسم الزبون'] in models['xgboost'][5] else 'عميل عادي'})\n"
            f"- المنتج: {row['اسم المنتج']}\n"
            f"- الشهر/السنة: {row['الشهر']}/{row['السنة']}\n"
            f"- رقم الحساب و تكرار العميل و المنتج\n"
            f"- { 'الصيف' if row['الشهر'] in [6,7,8,9] else 'خارج موسم الصيف'}\n"
            "قيم هذه العوامل مرت على النموذج وتعلم منها نمط الطلب والتغيرات."
        )
    elif model_name == "ARIMA":
        tips = (
            "- هذا النموذج مناسب أكثر للتوقعات الزمنية الشاملة (مثلاً التوزيع الشهري على مستوى ليبيا كلها).\n"
            "- أفضل دقة لما يكون فيه انتظام تاريخي واضح في الكميات.\n"
            "- لا يهتم بتفاصيل العميل أو المنتج، فقط التاريخ (السنة/الشهر) تؤثر في النتيجة."
        )
        why = (
            f"القيمة المتوقعة بناءً على بيانات الطلب الشهرية السابقة حتى تاريخ {row['الشهر']}/{row['السنة']}.\n"
            "لو الشهر أو السنة لم تكن موجودة في البيانات يدّيك متوسط أو تنبؤ مستقبلي بناءً على السلسلة الزمنية."
        )
    elif model_name == "ANN":
        tips = (
            "- النموذج العصبي يحلل العلاقات غير الخطية في كل العوامل.\n"
            "- حاول ملء كل الخانات بشكل صحيح لأن النموذج حساس لأي تغيير.\n"
            "- يفضل استخدامه لو عندك أنماط استهلاك غريبة أو معقدة ما تتفسرش بالسهولة."
        )
        why = (
            f"القيمة ناتجة عن معالجة كل المدخلات (العميل - المنتج - الشهر - كل العوامل المساعدة)\n"
            "الشبكة العصبية أحيانا تكتشف علاقات مخفية بين المتغيرات يصعب على أي موديل آخر اكتشافها."
        )
    else:
        tips = "اختر موديل للحصول على تفسير."
        why = "اختر موديل للحصول على تفسير."
    return tips, why

def analyze_prediction(pred_value, df_all, model_name):
    avg = df_all['الكمية باللتر'].mean()
    q1 = df_all['الكمية باللتر'].quantile(0.25)
    q3 = df_all['الكمية باللتر'].quantile(0.75)
    status, msg, color1, color2, icon = "", "", "", "", ""
    if pred_value > q3:
        status = "كمية مرتفعة 🚨"
        msg = "الكمية المتوقعة أعلى من أغلب الطلبات. تحقق من وجود حالة استثنائية أو طلب موسمي غير معتاد."
        color1, color2 = "#F85032", "#FFB347"
        icon = "🔴"
    elif pred_value < q1:
        status = "كمية منخفضة ℹ️"
        msg = "الكمية أقل من المعدل المعتاد. ربما يتعلق ذلك بانخفاض الطلب أو بيانات غير مكتملة."
        color1, color2 = "#00C9FF", "#92FE9D"
        icon = "🟢"
    else:
        status = "كمية ضمن النطاق 💡"
        msg = "الكمية المتوقعة تقع ضمن المعدل المعتاد. لا يوجد مؤشرات خطورة حالية."
        color1, color2 = "#43cea2", "#185a9d"
        icon = "🟠"
    return status, msg, color1, color2, icon

def main():
    st.set_page_config(page_title="توقع وتحليل كمية الديزل", layout="wide")
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

    st.sidebar.markdown("### رفع ملف إكسل خاص بك")
    excel_file = st.sidebar.file_uploader("ارفع ملف Excel بالهيكل نفسه (الأعمدة الأصلية):", type=["xlsx"])
    train_btn = st.sidebar.button("Train / إعادة تدريب الموديلات")

    if "user_df" not in st.session_state:
        st.session_state.user_df = None
    if "user_models" not in st.session_state:
        st.session_state.user_models = None

    if train_btn and excel_file:
        try:
            user_df = load_and_preprocess_data(excel_file)
            user_models = prepare_and_train_models(user_df)
            st.session_state.user_df = user_df
            st.session_state.user_models = user_models
            st.sidebar.success("تم رفع الملف وتدريب الموديلات على البيانات الجديدة بنجاح ✅")
        except Exception as e:
            st.sidebar.error(f"حدث خطأ أثناء تدريب الموديلات: {e}")

    if st.session_state.user_df is not None and st.session_state.user_models is not None:
        orig_df = st.session_state.user_df
        models = st.session_state.user_models
    else:
        orig_df = load_and_preprocess_data("ديزل 20-24.xlsx")
        models = prepare_and_train_models(orig_df)
        if orig_df is None:
            return

    page = st.sidebar.selectbox("اختر الصفحة", ["تحليل البيانات", "توقع الكمية", "أنسيت", "مقارنة النماذج"])

    if page == "تحليل البيانات":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(render_colored_card(
                "إجمالي الكمية", f"{int(models['xgboost'][6]['الكمية باللتر'].sum()):,}", "إجمالي الديزل الموزع",
                "#2196F3", "#1565C0", icon="🛢"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_colored_card(
                "عدد العملاء", models["xgboost"][6]["اسم الزبون"].nunique(), "عملاء فريدون",
                "#43A047", "#166534", icon="👤"), unsafe_allow_html=True)
        with col3:
            top_product_row = models["xgboost"][6].groupby('اسم المنتج')['الكمية باللتر'].sum().idxmax()
            top_product_val = int(models["xgboost"][6].groupby('اسم المنتج')['الكمية باللتر'].sum().max())
            st.markdown(render_colored_card(
                "الأكثر مبيعًا", f"{top_product_row}", f"{top_product_val:,} لتر",
                "#FFC107", "#FF6F00", icon="🏆"), unsafe_allow_html=True)
        with col4:
            top_customer_row = models["xgboost"][6].groupby('اسم الزبون')['الكمية باللتر'].sum().idxmax()
            top_customer_val = int(models["xgboost"][6].groupby('اسم الزبون')['الكمية باللتر'].sum().max())
            st.markdown(render_colored_card(
                "أكبر عميل", f"{top_customer_row}", f"{top_customer_val:,} لتر",
                "#EC407A", "#7B1FA2", icon="🥇"), unsafe_allow_html=True)
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            region_counts = models["xgboost"][6]['المنطقة'].value_counts()
            fig1 = px.pie(names=region_counts.index, values=region_counts.values, title='توزيع المناطق',
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig1.update_traces(textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
        with col_viz2:
            fig2 = px.histogram(models["xgboost"][6], x='الكمية باللتر', title='توزيع الكمية باللتر',
                               nbins=20, color_discrete_sequence=['#FF6F61'])
            fig2.update_layout(xaxis_title='الكمية باللتر', yaxis_title='عدد الصفوف')
            st.plotly_chart(fig2, use_container_width=True)
        monthly_avg = models["xgboost"][6].groupby('الشهر')['الكمية باللتر'].mean().reset_index()
        fig3 = px.line(monthly_avg, x='الشهر', y='الكمية باللتر', title='متوسط الكمية حسب الشهر',
                      line_shape='spline', color_discrete_sequence=['#43A047'])
        st.plotly_chart(fig3, use_container_width=True)
        top_customers = models["xgboost"][6].groupby('اسم الزبون')['الكمية باللتر'].sum().nlargest(10).reset_index()
        fig4 = px.bar(top_customers, x='اسم الزبون', y='الكمية باللتر', title='أعلى 10 عملاء حسب الكمية',
                     color_discrete_sequence=['#FFB300'])
        fig4.update_layout(xaxis_title='اسم الزبون', yaxis_title='الكمية باللتر', xaxis_tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)
        yearly_sum = models["xgboost"][6].groupby('السنة')['الكمية باللتر'].sum().reset_index()
        fig5 = px.line(yearly_sum, x='السنة', y='الكمية باللتر', title='إجمالي الكمية حسب السنة',
                      line_shape='spline', color_discrete_sequence=['#8E24AA'])
        st.plotly_chart(fig5, use_container_width=True)

    elif page == "توقع الكمية":
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        model_choice = st.selectbox("اختر الموديل", options=["XGBoost", "ARIMA", "ANN"])
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
                if model_choice == "XGBoost":
                    prediction, source = retrieve_or_predict(
                        new_data.iloc[0], orig_df, models["xgboost"][0], models["xgboost"][1], models["xgboost"][2],
                        models["xgboost"][3], models["xgboost"][4], models["xgboost"][5], models["xgboost"][6]
                    )
                elif model_choice == "ARIMA":
                    prediction, source = predict_arima(
                        new_data.iloc[0], models["arima"][0], models["arima"][1]
                    )
                elif model_choice == "ANN":
                    prediction, source = predict_ann(
                        new_data.iloc[0], models["ann"][0], models["ann"][1], models["ann"][2], models["ann"][3], models["ann"][4], models["ann"][5], models["ann"][6]
                    )
                prediction_file = save_prediction(new_data.iloc[0], prediction, source, model_choice)
                st.session_state.prediction_history.append({
                    **new_data.iloc[0],
                    'الكمية_المتوقعة_باللتر': prediction,
                    'مصدر': source,
                    'الموديل': model_choice
                })
                st.markdown(render_colored_card(
                    "🔥 النتيجة المتوقعة", f"{prediction:,.2f} لتر", f"المصدر: {source} - {model_choice}",
                    "#57C1EB", "#246FA8", radius="2.5rem", icon="⛽"
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

    elif page == "أنسيت":
        st.markdown(
            '<div style="font-size:2.0rem; font-weight:800; color:#2196F3; margin-bottom:1.2rem; text-align:center;">🧠 صفحة أنسيت - شرح التوقعات</div>',
            unsafe_allow_html=True
        )
        if 'prediction_history' in st.session_state and len(st.session_state.prediction_history) > 0:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            selected_index = st.selectbox(
                "اختر توقع لعرض تفاصيله:",
                options=list(range(len(history_df))),
                format_func=lambda i: f"{history_df.iloc[i]['الموديل']} | {history_df.iloc[i]['السنة']}/{history_df.iloc[i]['الشهر']} | {history_df.iloc[i]['اسم الزبون']}"
            )
            selected_pred = history_df.iloc[selected_index]
            status, msg, c1, c2, icon = analyze_prediction(selected_pred['الكمية_المتوقعة_باللتر'], models["xgboost"][6], selected_pred['الموديل'])
            st.markdown(f"""
                <div style='margin: 1.2rem 0 1.6rem 0; padding: 1.2rem 2.2rem 1.7rem 2.2rem; 
                            background: linear-gradient(90deg, {c1} 30%, {c2} 100%);
                            border-radius: 2.1rem; color: #fff; font-size:1.13rem; font-weight: 600; letter-spacing:1.1px; box-shadow:0 4px 32px 0 rgba(50,50,93,.09);'>
                <span style="font-size:1.15rem; font-weight:800; color:#fff;">تفصيل التوقع</span><br>
                الكمية المتوقعة: <span style="font-size:1.25rem;color:#fff;font-weight:900;">{selected_pred['الكمية_المتوقعة_باللتر']:.2f} لتر</span><br>
                الموديل المستخدم: <span style="font-size:1.05rem;font-weight:800;">{selected_pred['الموديل']}</span><br>
                المصدر: <span style="font-size:1.01rem;">{selected_pred['مصدر']}</span><br>
                <hr style='margin: 0.7rem 0 0.6rem 0; border-color: #fff;'>
                <span style="font-size:1.04rem;">{icon} <b>{status}</b></span>
                <br><span style="font-size:0.99rem;">{msg}</span>
                </div>
            """, unsafe_allow_html=True)
            tips, why = insight_explanation(selected_pred, selected_pred['الموديل'], models, orig_df)
            st.markdown(f"#### <span style='color:#1976D2;'>💡 نصائح هامة</span>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background: #192C3A; color:#d5e5f7; padding: 1rem 1.7rem; border-radius: 1.1rem; font-size:1.07rem;'>{tips.replace('-', '•')}</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"#### <span style='color:#F57C00;'>🤔 لماذا هذا التوقع؟</span>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background: #1A2636; color:#fff; padding: 1rem 1.7rem; border-radius: 1.1rem; font-size:1.07rem;'>{why.replace('-', '•')}</div>",
                unsafe_allow_html=True
            )
            st.markdown("### 📈 تحليلات رسومية للتوقع")
            import plotly.graph_objs as go
            d = models["xgboost"][6]
            pred_val = selected_pred['الكمية_المتوقعة_باللتر']
            selected_month = f"{selected_pred['السنة']}-{selected_pred['الشهر']}"
            customer_data = d[d['اسم الزبون'] == selected_pred['اسم الزبون']].sort_values(['السنة','الشهر'])
            st.markdown("#### 🔵 تطور طلبات هذا العميل عبر الزمن")
            if not customer_data.empty:
                time_labels = customer_data['السنة'].astype(str) + "-" + customer_data['الشهر'].astype(str)
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=time_labels,
                    y=customer_data['الكمية باللتر'],
                    mode='lines+markers',
                    name='طلبات العميل السابقة',
                    line=dict(color='rgba(50,140,255,0.8)', width=2),
                    marker=dict(size=10, color='rgba(12,45,130,0.85)', line=dict(width=2, color="#fff")),
                    hoverlabel=dict(bgcolor="#222", font=dict(color="#fff")),
                ))
                fig1.add_trace(go.Scatter(
                    x=[selected_month],
                    y=[pred_val],
                    mode='markers+text',
                    name="توقع الموديل",
                    marker=dict(size=18, color="orange", line=dict(width=4, color="#fff")),
                    text=["توقع"], textposition="top center"
                ))
                fig1.update_layout(
                    title="طلبات العميل الشهرية وتوقع الموديل",
                    xaxis_title="الشهر/السنة",
                    yaxis_title="الكمية باللتر",
                    plot_bgcolor="#232936",
                    paper_bgcolor="#232936",
                    font=dict(color="#F9FAFB", size=14),
                    xaxis=dict(tickangle=-45, showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#39414e"),
                    legend=dict(bgcolor="#222831", font=dict(size=13)),
                    height=320,
                    margin=dict(l=30, r=30, t=40, b=80),
                )
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown(
                    "<span style='color:#6AD49B; font-size:1.1rem;'>الرسم يوضح تطور كميات الطلبات الشهرية لهذا العميل بالتاريخ، والنقطة البرتقالية تمثل توقع النظام الحالي. لو التوقع بعيد عن الخط الأزرق بشكل ملحوظ، هذا معناه أن النموذج يتوقع تغير في الطلب مقارنة بالتاريخ.</span>",
                    unsafe_allow_html=True
                )
            else:
                st.info("العميل غير موجود في بيانات التدريب، لا يمكن عرض تاريخ الطلبات.")
            st.markdown("#### 🟢 تطور متوسط الكمية الشهرية لكل العملاء مع إبراز توقعك")
            monthly_avg = d.groupby(['السنة','الشهر'])['الكمية باللتر'].mean().reset_index()
            monthly_avg['تاريخ'] = monthly_avg['السنة'].astype(str) + "-" + monthly_avg['الشهر'].astype(str)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=monthly_avg['تاريخ'],
                y=monthly_avg['الكمية باللتر'],
                mode='lines+markers',
                name='متوسط الكميات الشهرية',
                line=dict(color='#36C1A2', width=3),
                marker=dict(size=8, color='#23353F', line=dict(width=2, color="#fff")),
                hoverlabel=dict(bgcolor="#222", font=dict(color="#fff")),
            ))
            fig2.add_trace(go.Scatter(
                x=[selected_month],
                y=[pred_val],
                mode="markers+text",
                name="توقعك الحالي",
                marker=dict(size=18, color="#FF8333", line=dict(width=4, color="#fff")),
                text=["توقع"], textposition="bottom center"
            ))
            fig2.update_layout(
                title="متوسط الكميات الشهرية لجميع العملاء",
                xaxis_title="الشهر/السنة",
                yaxis_title="متوسط الكمية باللتر",
                plot_bgcolor="#222831",
                paper_bgcolor="#222831",
                font=dict(color="#F9FAFB", size=14),
                xaxis=dict(tickangle=-45, showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#313C47"),
                legend=dict(bgcolor="#222831", font=dict(size=13)),
                height=320,
                margin=dict(l=30, r=30, t=40, b=80),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(
                "<span style='color:#A4CEFC; font-size:1.1rem;'>الرسم يعرض متوسط الكميات الشهرية لجميع العملاء، مع إبراز موقع التوقع الحالي. هذا يساعدك تشوف إذا كان توقعك أعلى أو أقل أو قريب من المعدل الطبيعي في السوق.</span>",
                unsafe_allow_html=True
            )
        else:
            st.warning("لا يوجد توقعات حتى الآن. ابدأ أولًا من صفحة 'توقع الكمية'.")

    elif page == "مقارنة النماذج":
        st.markdown(
            """
            <div style="background: linear-gradient(90deg,#43cea2 10%,#185a9d 90%);
                        border-radius:2rem;padding:1rem 2rem;margin-bottom:1.3rem;text-align:center;">
                <span style="font-size:2.2rem; font-weight:900; color:#fff;">🔎 مقارنة النماذج الثلاثة</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("### 📊 ملخص سريع")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style="background:#e3f1fb;border-radius:1.3rem;padding:1.25rem 1.1rem 1.2rem 1.1rem;">
            <b style='color:#1565C0;font-size:1.18rem;'>XGBoost</b><br>
            <span style='color:#223;'>• خوارزمية أشجار تعزيزية (Gradient Boosting Trees)<br>
            • قوية جدًا في التوقعات المعتمدة على خصائص متعددة.<br>
            • تصلح للبيانات التي تحتوي على علاقات معقدة وغير خطية.<br></span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background:#fff6e3;border-radius:1.3rem;padding:1.25rem 1.1rem 1.2rem 1.1rem;">
            <b style='color:#D84315;font-size:1.18rem;'>ARIMA</b><br>
            <span style='color:#332;'>• نموذج إحصائي للتنبؤ بالسلاسل الزمنية فقط.<br>
            • يعتمد على القيم السابقة للسلسلة.<br>
            • ممتاز لتوقع الطلب الكلي أو الموسمي عبر الوقت.<br></span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style="background:#f7f2ff;border-radius:1.3rem;padding:1.25rem 1.1rem 1.2rem 1.1rem;">
            <b style='color:#7B1FA2;font-size:1.18rem;'>ANN</b><br>
            <span style='color:#232;'>• شبكة عصبية اصطناعية (Artificial Neural Network).<br>ب
            • تتعلم الأنماط المخفية بين كل المتغيرات.<br>
            • ممتازة في حال وجود بيانات كبيرة وأنماط معقدة.<br></span>
            </div>
            """, unsafe_allow_html=True)


        st.markdown("---")
        st.markdown("### 🧮 المعادلات وفكرة العمل")
        with st.expander("🔹 معادلة/خوارزمية XGBoost"):
            st.latex(r"""
                \hat{y}_i = \sum_{k=1}^{K} f_k(x_i), \quad f_k \in \mathcal{F}
            """)
            st.markdown(
                "كل شجرة قرار (Tree) تتعلم تصحيح أخطاء الشجرة السابقة. النتيجة النهائية هي مجموع نتائج كل الأشجار."
            )
        with st.expander("🔸 معادلة ARIMA (التوقع الزمني)"):
            st.latex(r"""
                y_t = c + \phi_1 y_{t-1} + ... + \phi_p y_{t-p} + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q} + \epsilon_t
            """)
            st.markdown(
                "ARIMA يعتمد على القيم السابقة للسلسلة (y) وعلى التشويش (الضوضاء) لتوقع القيمة التالية."
            )
        with st.expander("🔺 معادلة ANN (الشبكة العصبية)"):
            st.latex(r"""
                \mathbf{y} = \sigma(\mathbf{W}_2 \cdot \sigma(\mathbf{W}_1 \cdot \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2)
            """)
            st.markdown(
                "الشبكة العصبية تتكون من طبقات متصلة، وكل طبقة تطبق تحويلات غير خطية على البيانات، وتتعلم من خلال التدريب."
            )

        st.markdown("---")
        st.markdown("### ⚖️ مميزات وعيوب كل نموذج")
        st.markdown("""
        | النموذج    | المميزات         | العيوب              |
        |:----------:|:----------------|:---------------------|
        | **XGBoost** | دقيق وسريع للبيانات الهيكلية، يتعامل مع علاقات معقدة، مقاوم لـ overfitting | يحتاج ضبط جيد للباراميترات، ليس الأمثل للبيانات الزمنية فقط |
        | **ARIMA**  | سهل التفسير، ممتاز للتسلسل الزمني، لا يحتاج خصائص كثيرة | لا يتعامل مع متغيرات خارجية، يحتاج بيانات تاريخية كافية |
        | **ANN**    | يتعلم أي علاقة مهما كانت معقدة، جيد للبيانات الضخمة | يحتاج بيانات كثيرة وتدريب قوي، أقل شفافية في التفسير  |
        """)

        st.markdown("---")
        st.markdown("### 📚 روابط الدوكمنتشن الرسمية")
        st.markdown("""
        - [🔗 XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
        - [🔗 ARIMA (statsmodels) Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
        - [🔗 Keras / ANN Documentation](https://keras.io/api/models/sequential/)
        """)

        st.info(
            "اختر النموذج المناسب حسب طبيعة بياناتك والغرض من التوقع. إذا أردت تفاصيل أكثر، اضغط على الروابط لقراءة المزيد."
        )

if __name__ == "__main__":
    main()
