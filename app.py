import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

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
  # 통화량 데이터 로드
  df_m1 = pd.read_csv('data/M1 상품별 구성내역(평잔, 원계열)_06092354.csv')
  df_m2 = pd.read_csv('data/M2 상품별 구성내역(평잔, 원계열)_06092535.csv')
  df_apt = pd.read_csv(
      'data/(월) 지역별 매매지수_아파트.csv', encoding='cp949'
  )

  # --- ECOS M1, M2 전처리 ---
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

  # --- 부동산원 아파트 매매지수 전처리 ---
  # '2006년 1월' 형태의 컬럼들을 '2006/01' 형태로 변환
  apt_date_cols = [c for c in df_apt.columns if '년' in c]

  # 지역 선택 (기본 '전국' 또는 '서울' 등)
  # 0번 행은 '지수' 메타데이터이므로 제외하고 실제 지역 행들만 추출
  df_apt_filtered = df_apt[df_apt['지역'] != '지역'].copy()

  return money_df, df_apt_filtered, apt_date_cols


try:
  money_df, apt_raw_df, apt_date_cols = load_all_data()
  st.success(
      '✅ 한국은행 통화량 원자료와 아파트 매매가격지수 데이터를 성공적으로'
      ' 불러왔습니다!'
  )

  # 사이드바 설정
  st.sidebar.header('⚙️ 분석 설정 및 필터')
  regions = apt_raw_df['지역'].unique().tolist()
  selected_region = st.sidebar.selectbox('지역 선택', regions, index=0)

  # 아파트 데이터를 롱 포맷(Long format)으로 변환
  region_row = apt_raw_df[apt_raw_df['지역'] == selected_region]
  apt_melted_list = []
  for col in apt_date_cols:
    val = region_row[col].values[0]
    # 'YYYY년 M월'을 'YYYY/MM'으로 변환
    parts = col.replace('년 ', '/').replace('월', '').strip()
    year, month = parts.split('/')
    formatted_date = f'{year}/{int(month):02d}'
    apt_melted_list.append(
        {'연월': formatted_date, '아파트매매지수': float(str(val).replace(',', ''))}
    )

  apt_df_clean = pd.DataFrame(apt_melted_list)

  # 통화량 데이터와 아파트 지수 데이터 병합
  merged_df = pd.merge(money_df, apt_df_clean, on='연월', how='inner').dropna()

  # 기간 선택 슬라이더
  all_months = merged_df['연월'].tolist()
  selected_range = st.sidebar.select_slider(
      '분석 조회 기간 선택',
      options=all_months,
      value=(all_months[0], all_months[-1]),
  )

  start_m, end_m = selected_range
  filtered_df = merged_df[
      (merged_df['연월'] >= start_m) & (merged_df['연월'] <= end_m)
  ].reset_index(drop=True)

  # ==========================================
  # PART 1: M1 통화량 단독 분석 (위기 및 단기 유동성)
  # ==========================================
  st.markdown('---')
  st.header(
      '🚨 PART 1: M1 통화량 단독 분석 (위기 상황 및 단기 유동성·자금 경색 감지)'
  )
  st.markdown(
      'M1(협의통화)은 경제 위기나 신용 경색(IMF, 금융위기 등) 시 가계와'
      ' 기업의 자금 대이동 및 단기 유동성 변화를 가장 민감하게 보여줍니다.'
  )

  fig_m1, ax_m1 = plt.subplots(figsize=(12, 4))
  ax_m1.plot(
      filtered_df['연월'],
      filtered_df['M1_금액'],
      color='tab:red',
      linewidth=2,
      label='M1 통화량 (평잔, 십억원)',
  )
  ax_m1.set_title(
      f'M1 단독 추이 ({start_m} ~ {end_m})', fontsize=13, fontweight='bold'
  )
  ax_m1.set_xlabel('연월')
  ax_m1.set_ylabel('M1 통화량 (십억원)', color='tab:red')
  step_m1 = max(1, len(filtered_df) // 12)
  ax_m1.set_xticks(filtered_df['연월'][::step_m1])
  plt.xticks(rotation=45)
  ax_m1.grid(True, linestyle='--', alpha=0.5)
  ax_m1.legend(loc='upper left')
  fig_m1.tight_layout()
  st.pyplot(fig_m1)

  # ==========================================
  # PART 2: M2 통화량 & 아파트 가격 비교 및 회귀분석
  # ==========================================
  st.markdown('---')
  st.header(
      f'🏠 PART 2: M2 통화량 vs 아파트 매매가격지수 ({selected_region}) 회귀분석'
      ' 및 미래 예측'
  )
  st.markdown(
      '아파트 같은 중장기 실물 자산 가격은 시중에 풀린 총누적 유동성인'
      ' **M2**와 가장 밀접한 상관관계를 가집니다.'
  )

  fig_m2, ax1 = plt.subplots(figsize=(12, 5))

  # M2 그래프 (좌측 축)
  ax1.plot(
      filtered_df['연월'],
      filtered_df['M2_금액'],
      color='tab:blue',
      linewidth=2,
      label='M2 통화량 (십억원)',
  )
  ax1.set_xlabel('연월', fontsize=12)
  ax1.set_ylabel('M2 통화량 (십억원)', color='tab:blue', fontsize=12)
  ax1.tick_params(axis='y', labelcolor='tab:blue')
  step_m2 = max(1, len(filtered_df) // 12)
  ax1.set_xticks(filtered_df['연월'][::step_m2])
  plt.xticks(rotation=45)

  # 아파트 가격지수 그래프 (우측 축)
  ax2 = ax1.twinx()
  ax2.plot(
      filtered_df['연월'],
      filtered_df['아파트매매지수'],
      color='tab:green',
      linewidth=2,
      linestyle='-',
      label=f'아파트 매매가격지수 ({selected_region})',
  )
  ax2.set_ylabel(
      f'아파트 매매가격지수 ({selected_region})', color='tab:green', fontsize=12
  )
  ax2.tick_params(axis='y', labelcolor='tab:green')

  plt.title(
      f'M2 통화량과 {selected_region} 아파트 가격지수 장기 추세 비교',
      fontsize=14,
      fontweight='bold',
  )
  fig_m2.tight_layout()
  st.pyplot(fig_m2)

  # 회귀분석 (M2를 독립변수 X, 아파트 지수를 종속변수 y로 설정)
  st.subheader('🤖 M2 기반 아파트 가격지수 선형회귀분석 및 예측 시뮬레이션')

  X = filtered_df[['M2_금액']].values
  y = filtered_df['아파트매매지수'].values

  model = LinearRegression()
  model.fit(X, y)
  r2 = model.score(X, y)

  st.info(
      f'📌 **회귀 모델 성능 (결정계수 R²): {r2:.4f}** (M2 통화량이 아파트'
      ' 가격지수 변동을 설명하는 설명력)'
  )

  # 미래 M2 통화량 슬라이더 예측
  min_m2 = float(filtered_df['M2_금액'].min())
  max_m2 = float(filtered_df['M2_금액'].max()) * 1.4

  future_m2_input = st.slider(
      '🔮 미래 예상 M2 통화량 입력 슬라이더 (십억원)',
      min_value=min_m2,
      max_value=max_m2,
      value=float(filtered_df['M2_금액'].max()),
      step=10000.0,
  )

  predicted_index = model.predict([[future_m2_input]])[0]
  st.success(
      f'✨ 예상 미래 M2 통화량(**{future_m2_input:,.0f} 십억원**)일 때, 회귀식으로'
      f' 추정되는 **{selected_region}** 아파트 예상 매매가격지수는'
      f' **{predicted_index:,.2f}** 입니다.'
  )

except Exception as e:
  st.error(
      f'데이터 처리 중 오류가 발생했습니다. 파일명과 컬럼 구조를 확인해 주세요.'
      f' 상세 에러: {e}'
  )
