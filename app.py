import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

st.set_page_config(
    page_title="통화량(M1, M2)에 따른 아파트 가격 추세 및 예측", layout="wide"
)

st.title("📈 통화량(M1, M2) & 아파트 가격 회귀분석 및 예측 프로그램")
st.markdown(
    "본 프로그램은 **한국은행(ECOS) 공식 통화량 원자료**와 **아파트 가격(지수)"**"
    " 데이터를 연동하여 분석합니다."
)


# ECOS CSV 파일 파싱 함수
def parse_ecos_csv(uploaded_file):
  df = pd.read_csv(uploaded_file)
  date_cols = [c for c in df.columns if '/' in c]
  melted = df.melt(
      id_vars=['통계표', '계정항목', '단위', '변환'],
      value_vars=date_cols,
      var_name='연월',
      value_name='금액',
  )
  melted['금액'] = (
      melted['금액'].astype(str).str.replace(',', '').str.strip().astype(float)
  )
  melted['연월'] = melted['연월'].str.replace(' p)', '').str.strip()
  item_name = melted['계정항목'].iloc[0]
  pivot_df = melted.pivot(index='연월', columns='계정항목', values='금액').reset_index()
  return pivot_df


# 사이드바: 데이터 업로드
st.sidebar.header("📁 데이터 파일 업로드")

# 기본 제공되는 파일 경로 활용 또는 추가 업로드 가능
m1_file = st.sidebar.file_uploader(
    "M1 통화량 파일 업로드", type=['csv'], key='m1'
)
m2_file = st.sidebar.file_uploader(
    "M2 통화량 파일 업로드", type=['csv'], key='m2'
)
apt_file = st.sidebar.file_uploader(
    "아파트 가격 데이터 파일 (CSV 또는 Excel)", type=['csv', 'xlsx'], key='apt'
)

# 파일이 업로드된 경우 처리
if m1_file and m2_file and apt_file:
  df_m1 = parse_ecos_csv(m1_file)
  df_m2 = parse_ecos_csv(m2_file)

  if apt_file.name.endswith('.csv'):
    df_apt = pd.read_csv(apt_file)
  else:
    df_apt = pd.read_excel(apt_file)

  st.success('✅ 모든 데이터가 성공적으로 로드되었습니다!')

  # 데이터 미리보기
  with st.expander('🔍 원본 데이터 확인하기'):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
      st.write('M1 데이터 샘플', df_m1.head(3))
    with col_b:
      st.write('M2 데이터 샘플', df_m2.head(3))
    with col_c:
      st.write('아파트 가격 데이터 샘플', df_apt.head(3))

  # 데이터 병합 (연월 기준)
  # 주의: 아파트 가격 데이터프레임에 '연월' (예: '2020/01') 및 가격 컬럼명이 존재해야 합니다.
  try:
    # M1, M2 병합
    m_merged = pd.merge(df_m1, df_m2, on='연월')
    # 아파트 데이터와 병합 (아파트 데이터의 날짜 컬럼명과 가격 컬럼명을 맞춰야 함)
    # 예시 코드: df_apt의 날짜 컬럼이 '연월', 가격 컬럼이 '가격' 또는 '지수'라고 가정
    # 사용자 환경에 맞게 컬럼명을 선택할 수 있도록 UI 제공
    st.markdown('---')
    st.subheader('🛠️ 분석 변수 선택')
    apt_cols = [c for c in df_apt.columns if c != '연월']
    selected_apt_col = st.selectbox(
        '아파트 가격(또는 지수)에 해당하는 컬럼 선택', apt_cols
    )
    date_col_apt = st.selectbox(
        '아파트 데이터의 연월(날짜) 컬럼 선택', df_apt.columns
    )

    # 데이터 표준화 정제
    df_apt_clean = df_apt[[date_col_apt, selected_apt_col]].copy()
    df_apt_clean.columns = ['연월', '아파트가격']
    df_apt_clean['연월'] = df_apt_clean['연월'].astype(str).str.strip()

    final_df = pd.merge(m_merged, df_apt_clean, on='연월').dropna()

    # 통화량 컬럼명 확인
    m1_col_name = [c for c in df_m1.columns if c != '연월'][0]
    m2_col_name = [c for c in df_m2.columns if c != '연월'][0]

    st.subheader('📊 통화량(M2) vs 아파트 가격 추세 그래프')
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        final_df['연월'],
        final_df[m2_col_name],
        color='tab:blue',
        label=f'M2 통화량 ({m2_col_name})',
    )
    ax1.set_xlabel('연월')
    ax1.set_ylabel('M2 통화량 (십억원)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(final_df['연월'][::12])  # 1년 단위로 눈금 표시
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(
        final_df['연월'],
        final_df['아파트가격'],
        color='tab:red',
        label='아파트 가격/지수',
    )
    ax2.set_ylabel('아파트 가격/지수', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('연도별 통화량과 아파트 가격 추이 비교')
    st.pyplot(fig)

    # 회귀분석 수행 (예: M2 통화량을 독립변수 X, 아파트 가격을 종속변수 y로 설정)
    st.markdown('---')
    st.subheader('🤖 선형회귀분석(Linear Regression) 및 미래 예측')

    X = final_df[[m2_col_name]].values
    y = final_df['아파트가격'].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    st.info(
        f'회귀 모델 결정계수(R² Score): **{r2:.4f}** (통화량이 아파트 가격 변동을'
        ' 설명하는 정도)'
    )

    # 미래 통화량 입력 슬라이더
    min_m2_val = float(final_df[m2_col_name].min())
    max_m2_val = float(final_df[m2_col_name].max()) * 1.5

    future_m2 = st.slider(
        f'미래 예상 {m2_col_name} (십억원)',
        min_value=min_m2_val,
        max_value=max_m2_val,
        value=float(final_df[m2_col_name].max()),
        step=10000.0,
    )

    predicted_apt_price = model.predict([[future_m2]])[0]
    st.success(
        f'🔮 예상 미래 통화량(**{future_m2:,.0f} 십억원**)일 때, 예측된 아파트'
        f' 가격(지수)은 **{predicted_apt_price:,.2f}** 입니다.'
    )

  except Exception as e:
    st.error(
        f'데이터 병합 또는 분석 중 오류가 발생했습니다. 컬럼 형식을 확인해주세요.'
        f' 상세 에러: {e}'
    )

else:
  st.warning(
      '⚠️ 사이드바에서 M1, M2 파일과 아파트 가격 데이터를 모두 업로드해 주세요.'
  )
  st.markdown(
      '**💡 팁:** 아파트 가격 데이터는 한국부동산원(R-ONE)의 [아파트 매매가격지수]'
      ' 시계열 데이터를 엑셀로 다운로드하여 사용하시면 가장 편리합니다.'
  )
