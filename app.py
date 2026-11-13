from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

# --- 폰트 깨짐 방지 설정 ---
import platform

if platform.system() == 'Windows':
  plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
  plt.rc('font', family='AppleGothic')
else:
  plt.rc('font', family='DejaVu Sans')

plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(
    page_title='통화량 및 아파트 가격 분석 프로그램', layout='wide'
)

st.title(
    '📊 대한민국 통화량(M1, M2)과 아파트 가격 선형회귀 분석 및 예측 시스템'
)
st.markdown(
    '한국은행(ECOS) 공식 통화량 데이터와 한국부동산원 아파트 매매가격지수를'
    ' 활용합니다.'
)


# 데이터 로드 및 전처리 함수
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

  region_en_map = {'전국': 'Nationwide', '서울': 'Seoul'}
  region_plot_name = region_en_map.get(selected_region, selected_region)

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

  # 사이드바 캘린더 날짜 선택기
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
    st.header(
        '🚨 PART 1: M1 통화량 단독 분석 (위기 상황 및 단기 유동성·자금 경색 감지)'
    )
    st.markdown(
        'M1(협의통화)은 경제 위기나 신용 경색 시 가계와 기업의 단기 유동성'
        ' 변화를 가장 민감하게 보여줍니다.'
    )

    fig_m1, ax_m1 = plt.subplots(figsize=(12, 4))
    ax_m1.plot(
        filtered_df['연월'],
        filtered_df['M1_금액'],
        color='tab:red',
        linewidth=2,
        label='M1 Supply (Billions KRW)',
    )
    ax_m1.set_title(
        f'M1 Trend Analysis ({start_m} ~ {end_m})', fontsize=13, fontweight='bold'
    )
    ax_m1.set_xlabel('Date (Year/Month)')
    ax_m1.set_ylabel('M1 Amount (Billions KRW)', color='tab:red')
    step_m1 = max(1, len(filtered_df) // 12)
    ax_m1.set_xticks(filtered_df['연월'][::step_m1])
    plt.xticks(rotation=45)
    ax_m1.grid(True, linestyle='--', alpha=0.5)
    ax_m1.legend(loc='upper left')
    fig_m1.tight_layout()
    st.pyplot(fig_m1)

    # PART 2: M2 통화량 & 아파트 가격 비교
    st.markdown('---')
    st.header(
        f'🏠 PART 2: M2 통화량 vs 아파트 매매가격지수 ({selected_region})'
        ' 회귀분석 및 예측'
    )
    st.markdown(
        '아파트 같은 중장기 실물 자산 가격은 시중에 풀린 총누적 유동성인'
        ' **M2**와 가장 밀접한 상관관계를 가집니다.'
    )

    fig_m2, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(
        filtered_df['연월'],
        filtered_df['M2_금액'],
        color='tab:blue',
        linewidth=2,
        label='M2 Supply (Billions KRW)',
    )
    ax1.set_xlabel('Date (Year/Month)', fontsize=12)
    ax1.set_ylabel('M2 Amount (Billions KRW)', color='tab:blue', fontsize=12)
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
        label=f'Apt Price Index ({region_plot_name})',
    )
    ax2.set_ylabel(
        f'Apt Price Index ({region_plot_name})',
        color='tab:green',
        fontsize=12,
    )
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title(
        f'M2 Supply & {region_plot_name} Apt Price Index Trend ({start_m} ~'
        f' {end_m})',
        fontsize=14,
        fontweight='bold',
    )
    fig_m2.tight_layout()
    st.pyplot(fig_m2)

    # 회귀분석 및 슬라이더 예측 시뮬레이션 (슬라이더 연동 산점도/회귀선 그래프 추가)
    st.markdown('---')
    st.subheader(
        '🤖 M2 기반 아파트 가격지수 선형회귀분석 및 실시간 예측 시뮬레이션'
    )

    X = filtered_df[['M2_금액']].values
    y = filtered_df['아파트매매지수'].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    st.info(
        f'📌 **회귀 모델 성능 (결정계수 R²): {r2:.4f}** (선택 기간 내 M2'
        ' 통화량이 아파트 가격지수 변동을 설명하는 설명력)'
    )

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

    # [추가됨] 슬라이더 조절 시 실시간으로 움직이는 M2 vs 아파트지수 회귀 예측 산점도 그래프
    st.markdown('#### 📈 M2 통화량 vs 아파트 가격지수 회귀 직선 및 예측 위치 시각화')

    fig_reg, ax_reg = plt.subplots(figsize=(10, 5))
    # 실제 데이터 산점도
    ax_reg.scatter(
        filtered_df['M2_금액'],
        filtered_df['아파트매매지수'],
        color='tab:blue',
        alpha=0.6,
        label='Historical Data',
    )

    # 회귀 직선 그리기
    x_line = np.linspace(min_m2, max_m2, 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax_reg.plot(
        x_line,
        y_line,
        color='tab:red',
        linewidth=2,
        label='Linear Regression Fit',
    )

    # 사용자가 슬라이더로 조절한 미래 예측 지점 표시 (움직일 때마다 빨간 점이 이동함)
    ax_reg.scatter(
        [future_m2_input],
        [predicted_index],
        color='gold',
        s=150,
        zorder=5,
        edgecolors='black',
        label=f'Predicted Point ({future_m2_input:,.0f}, {predicted_index:.2f})',
    )

    ax_reg.set_xlabel('M2 Supply (Billions KRW)', fontsize=11)
    ax_reg.set_ylabel(
        f'Apt Price Index ({region_plot_name})', fontsize=11, color='tab:green'
    )
    ax_reg.set_title(
        f'Regression Model & Interactive Prediction for {region_plot_name}',
        fontsize=13,
        fontweight='bold',
    )
    ax_reg.grid(True, linestyle='--', alpha=0.5)
    ax_reg.legend(loc='upper left')
    fig_reg.tight_layout()
    st.pyplot(fig_reg)

    # 분석 리포트 CSV 다운로드 기능
    st.markdown('---')
    st.subheader('📥 분석 결과 리포트 다운로드')
    st.markdown(
        '현재 조회된 기간과 계산된 통화량·아파트 지수 데이터를 CSV 파일로'
        ' 다운로드할 수 있습니다.'
    )

    csv_data = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label='📄 현재 분석 데이터 CSV 다운로드',
        data=csv_data,
        file_name=f'apartment_liquidity_report_{selected_region}.csv',
        mime='text/csv',
    )

except Exception as e:
  st.error(f'데이터 처리 중 오류가 발생했습니다: {e}')
