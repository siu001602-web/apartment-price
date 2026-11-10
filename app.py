from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# --- 폰트 깨짐(두부 현상/가나가 글자로 뜨는 현상) 방지 설정 ---
import platform

if platform.system() == 'Windows':
  plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
  plt.rc('font', family='AppleGothic')
else:
  # Linux (Streamlit Cloud 환경)에서는 산세리프 기본 폰트 사용 및 마이너스 기호 깨짐 방지
  plt.rc('font', family='DejaVu Sans')

plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(
    page_title='통화량 및 아파트 가격 분석 프로그램', layout='wide'
)

st.title('📊 대한민국 통화량(M1, M2)과 아파트 가격 분석 및 예측 프로그램')
st.markdown(
    '한국은행(ECOS) 공식 통화량 데이터와 한국부동산원 아파트 매매가격지수를'
    ' 활용합니다.'
)


# 1. 데이터 로드 및 전처리 함수
@st.cache_data
def load_all_data():
  df_m1 = pd.read_csv('data/M1 상품별 구성내역(평잔, 원계열)_06092354.csv')
  df_m2 = pd.read_csv('data/M2 상품별 구성내역(평잔, 원계열)_06092535.csv')
  df_apt = pd.read_csv(
      'data/(월) 지역별 매매지수_아파트.csv', encoding='cp949'
  )

  date_cols = [c for c in df_m1.columns if '/' in c]

  m1_melted = df_m1.melt(
      id_vars=['통계표', '계정항목', '단위', '변환'],
      value_vars=date_cols,
      var_name='연월',
      value_name='M1_금액',
  )
  m2_melted = df_m2.melt(
      id_vars=['통계표', '계정항목', '단위', '변환'],
      value_vars=date_cols,
      var_name='연월',
      value_name='M2_금액',
  )

  m1_melted['M1_금액'] = (
      m1_melted['M1_금액']
      .astype(str)
      .str.replace(',', '')
      .str.strip()
      .astype(float)
  )
  m2_melted['M2_금액'] = (
      m2_melted['M2_금액']
      .astype(str)
      .str.replace(',', '')
      .str.strip()
      .astype(float)
  )

  m1_melted['연월'] = m1_melted['연월'].str.replace(' p)', '').str.strip()
  m2_melted['연월'] = m2_melted['연월'].str.replace(' p)', '').str.strip()

  money_df = pd.merge(
      m1_melted[['연월', 'M1_금액']], m2_melted[['연월', 'M2_금액']], on='연월'
  )

  apt_date_cols = [c for c in df_apt.columns if '년' in c]
  df_apt_filtered = df_apt[df_apt['지역'] != '지역'].copy()

  return money_df, df_apt_filtered, apt_date_cols


try:
  money_df, apt_raw_df, apt_date_cols = load_all_data()
  st.success('✅ 데이터 로드 완료!')

  st.sidebar.header('⚙️ 분석 설정 및 필터')
  regions = apt_raw_df['지역'].unique().tolist()
  selected_region = st.sidebar.selectbox('지역 선택', regions, index=0)

  region_row = apt_raw_df[apt_raw_df['지역'] == selected_region]
  apt_melted_list = []
  for col in apt_date_cols:
    val = region_row[col].values[0]
    parts = col.replace('년 ', '/').replace('월', '').strip()
    year, month = parts.split('/')
    formatted_date = f'{year}/{int(month):02d}'
    apt_melted_list.append(
        {'연월': formatted_date, '아파트매매지수': float(str(val).replace(',', ''))}
    )

  apt_df_clean = pd.DataFrame(apt_melted_list)
  merged_df = pd.merge(money_df, apt_df_clean, on='연월', how='inner').dropna()

  # 캘린더 형태의 날짜 선택기
  st.sidebar.subheader('📅 조회 기간 선택 (캘린더)')

  min_ym = merged_df['연월'].min()
  max_ym = merged_df['연월'].max()

  min_date = datetime.strptime(min_ym, '%Y/%m').date()
  max_date = datetime.strptime(max_ym, '%Y/%m').date()

  start_date_input = st.sidebar.date_input(
      '시작 월', value=min_date, min_value=min_date, max_value=max_date
  )
  end_date_input = st.sidebar.date_input(
      '종료 월', value=max_date, min_value=min_date, max_value=max_date
  )

  start_m = start_date_input.strftime('%Y/%m')
  end_m = end_date_input.strftime('%Y/%m')

  if start_m > end_m:
    st.sidebar.error('⚠️ 시작 월이 종료 월보다 이후일 수 없습니다.')
    start_m, end_m = min_ym, max_ym

  filtered_df = merged_df[
      (merged_df['연월'] >= start_m) & (merged_df['연월'] <= end_m)
  ].reset_index(drop=True)

  if filtered_df.empty:
    st.warning('⚠️ 선택하신 기간에 해당하는 데이터가 없습니다.')
  else:
    # PART 1: M1 통화량 단독 분석
    st.markdown('---')
    st.header('🚨 PART 1: M1 Money Supply Analysis (Short-term Liquidity)')

    fig_m1, ax_m1 = plt.subplots(figsize=(12, 4))
    ax_m1.plot(
        filtered_df['연월'],
        filtered_df['M1_금액'],
        color='tab:red',
        linewidth=2,
        label='M1 Supply (Billions KRW)',
    )
    ax_m1.set_title(f'M1 Trend ({start_m} ~ {end_m})', fontsize=13)
    ax_m1.set_xlabel('Date')
    ax_m1.set_ylabel('M1 (Billions KRW)', color='tab:red')
    step_m1 = max(1, len(filtered_df) // 12)
    ax_m1.set_xticks(filtered_df['연월'][::step_m1])
    plt.xticks(rotation=45)
    ax_m1.grid(True, linestyle='--', alpha=0.5)
    ax_m1.legend(loc='upper left')
    fig_m1.tight_layout()
    st.pyplot(fig_m1)

    # PART 2: M2 통화량 & 아파트 가격 비교 및 회귀분석
    st.markdown('---')
    st.header(
        f'🏠 PART 2: M2 vs Apartment Price Index ({selected_region})'
        ' Regression'
    )

    fig_m2, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        filtered_df['연월'],
        filtered_df['M2_금액'],
        color='tab:blue',
        linewidth=2,
        label='M2 Supply (Billions KRW)',
    )
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('M2 (Billions KRW)', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    step_m2 = max(1, len(filtered_df) // 12)
    ax1.set_xticks(filtered_df['연월'][::step_m2])
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(
        filtered_df['연월'],
        filtered_df['아파트매매지수'],
        color='tab:green',
        linewidth=2,
        label=f'Apt Price Index ({selected_region})',
    )
    ax2.set_ylabel(
        f'Apt Price Index ({selected_region})', color='tab:green', fontsize=12
    )
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title(
        f'M2 Supply & {selected_region} Apt Price Index Trend ({start_m} ~'
        f' {end_m})',
        fontsize=14,
    )
    fig_m2.tight_layout()
    st.pyplot(fig_m2)

    # 회귀분석
    st.subheader(
        '🤖 Linear Regression & Future Price Prediction based on M2'
    )

    X = filtered_df[['M2_금액']].values
    y = filtered_df['아파트매매지수'].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    st.info(f'📌 **Regression Model R² Score: {r2:.4f}**')

    min_m2 = float(filtered_df['M2_금액'].min())
    max_m2 = float(filtered_df['M2_금액'].max()) * 1.4

    future_m2_input = st.slider(
        '🔮 Future M2 Supply Input Slider (Billions KRW)',
        min_value=min_m2,
        max_value=max_m2,
        value=float(filtered_df['M2_금액'].max()),
        step=10000.0,
    )

    predicted_index = model.predict([[future_m2_input]])[0]
    st.success(
        f'✨ Predicted Apt Price Index for **{selected_region}** at M2'
        f' **{future_m2_input:,.0f} Billion KRW**: **{predicted_index:,.2f}**'
    )

except Exception as e:
  st.error(f'데이터 처리 중 오류가 발생했습니다: {e}')
