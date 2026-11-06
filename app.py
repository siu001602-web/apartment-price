import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title='통화량(M1, M2) 데이터 분석 프로그램', layout='wide'
)

st.title('📊 대한민국 통화량(M1, M2) 추세 분석 프로그램')
st.markdown(
    'GitHub의 `data/` 폴더에 있는 한국은행 원자료를 불러와서 시각화합니다.'
)


# 한국은행 CSV 파일을 파싱하는 함수
@st.cache_data
def load_data():
  # 파일 경로 지정
  m1_path = 'data/M1 상품별 구성내역(평잔, 원계열)_06092354.csv'
  m2_path = 'data/M2 상품별 구성내역(평잔, 원계열)_06092535.csv'

  df_m1 = pd.read_csv(m1_path)
  df_m2 = pd.read_csv(m2_path)

  # 날짜 컬럼 추출 ('/'가 포함된 컬럼)
  date_cols = [c for c in df_m1.columns if '/' in c]

  # 데이터 가로 형태(Wide)를 세로 형태(Long)로 변환
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

  # 숫자 형식 변환 (콤마 제거 및 실수형 변환)
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

  # 연월 정리 (' p)' 등 불필요한 문자 제거)
  m1_melted['연월'] = m1_melted['연월'].str.replace(' p)', '').str.strip()
  m2_melted['연월'] = m2_melted['연월'].str.replace(' p)', '').str.strip()

  # M1과 M2 데이터 병합
  merged_df = pd.merge(
      m1_melted[['연월', 'M1_금액']], m2_melted[['연월', 'M2_금액']], on='연월'
  )
  return merged_df


# 데이터 로드 실행
try:
  df = load_data()
  st.success('✅ M1, M2 원자료를 성공적으로 불러왔습니다!')

  # 데이터 미리보기
  with st.expander('🔍 불러온 데이터 미리보기'):
    st.dataframe(df.tail(10))  # 가장 최신 데이터 10개 표시

  # 기본 통계 정보
  st.subheader('📈 통화량 요약 지표')
  col1, col2 = st.columns(2)
  with col1:
    st.metric(
        label='최신 M1 통화량 (십억원)',
        value=f'{df["M1_금액"].iloc[-1]:,.1f}',
        delta=f'{df["M1_금액"].iloc[-1] - df["M1_금액"].iloc[-2]:,.1f}',
    )
  with col2:
    st.metric(
        label='최신 M2 통화량 (십억원)',
        value=f'{df["M2_금액"].iloc[-1]:,.1f}',
        delta=f'{df["M2_금액"].iloc[-1] - df["M2_금액"].iloc[-2]:,.1f}',
    )

  # 시각화 그래프 그리기
  st.subheader('📉 M1 및 M2 통화량 추이 그래프')

  fig, ax1 = plt.subplots(figsize=(12, 5))

  # M2 그래프 (좌측 축)
  ax1.plot(
      df['연월'],
      df['M2_금액'],
      color='tab:blue',
      label='M2 통화량 (십억원)',
      linewidth=2,
  )
  ax1.set_xlabel('연월', fontsize=12)
  ax1.set_ylabel('M2 통화량 (십억원)', color='tab:blue', fontsize=12)
  ax1.tick_params(axis='y', labelcolor='tab:blue')

  # 1년(12개월) 단위로 X축 눈금 표시 설정
  ax1.set_xticks(df['연월'][::12])
  plt.xticks(rotation=45)

  # M1 그래프 (우측 축)
  ax2 = ax1.twinx()
  ax2.plot(
      df['연월'],
      df['M1_금액'],
      color='tab:orange',
      label='M1 통화량 (십억원)',
      linewidth=2,
      linestyle='--',
  )
  ax2.set_ylabel('M1 통화량 (십억원)', color='tab:orange', fontsize=12)
  ax2.tick_params(axis='y', labelcolor='tab:orange')

  plt.title('한국은행 M1 및 M2 통화량 장기 추세 비교', fontsize=14)
  fig.tight_layout()

  st.pyplot(fig)

except Exception as e:
  st.error(
      f'데이터를 처리하는 동안 오류가 발생했습니다. 파일 경로와 형식을'
      f' 확인해주세요. 상세 에러: {e}'
  )
