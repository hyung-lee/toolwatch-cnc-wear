import json,time
import numpy as np,pandas as pd,streamlit as st
from PIL import Image
import joblib
from core import ROOT,load_model,predict,history_features
st.set_page_config(page_title='TOOLWATCH | 공구 마모 모니터',page_icon='🔧',layout='wide')
st.markdown('''<style>.stApp{background:#0b1220;color:#edf2fa}div[data-testid="stMetric"]{background:#172238;padding:18px;border-radius:12px}h1{letter-spacing:-1px}</style>''',unsafe_allow_html=True)
st.title('🔧 TOOLWATCH · 공구 마모 모니터')
st.caption('Nonastreda 실제 공개 이미지로 학습 · 공구 ID 단위 분리 · Python / PyTorch / OpenCV / Streamlit')
if not (ROOT/'metrics.json').exists():
 st.error('학습된 모델이 없습니다. python download.py → python train.py 실행 후 다시 열어주세요.');st.stop()
@st.cache_resource
def resources():return (*load_model(),joblib.load(ROOT/'rul.joblib'))
m,ck,rul=resources();report=json.loads((ROOT/'metrics.json').read_text());df=pd.read_csv(ROOT/'predictions.csv')
with st.sidebar:
 st.header('측정 설정');threshold=st.number_input('교체 검토 기준 (µm)',min_value=1.,value=150.,step=10.)
 st.caption('150 µm는 시연용 사용자 설정값입니다. 실제 공구·공정별 허용 기준을 적용하세요.')
 session=st.text_input('공구 / 날 이력 이름','upload-tool-1-blade-1')
 st.caption('같은 공구의 같은 날을 같은 배율·조명에서 촬영하세요.')
 if st.button('이 공구 이력 초기화'):st.session_state['hist_'+session]=[]
mode=st.radio('입력 방식',['이미지 업로드','공개 테스트 데이터 시연'],horizontal=True)
im=None;hist=[]
if mode=='공개 테스트 데이터 시연':
 test=df[df.split=='test'];tool=st.selectbox('미학습 공구',sorted(test.tool.unique()));blade=st.selectbox('날 번호',sorted(test[test.tool==tool].blade.unique()));g=test[(test.tool==tool)&(test.blade==blade)].sort_values('run')
 run=st.select_slider('가공 회차',options=g.run.tolist());current=g[g.run==run].iloc[0];im=Image.open(ROOT/current.tool_file_name)
 hist=[{'run':float(r.run),'wear':float(r.predicted_wear)} for r in g[g.run<=run].itertuples()]
 st.caption('이전 회차까지의 예측을 재생합니다. T9·T10 사진은 이미지 모델 학습과 검증에 사용하지 않았습니다.')
else:
 key='hist_'+session;hist=st.session_state.setdefault(key,[])
 run=st.number_input('가공 회차 (증가하는 값)',min_value=0.,value=float(hist[-1]['run']+1) if hist else 1.,step=1.)
 if mode=='이미지 업로드':
  st.caption('사진을 올리면 즉시 마모량을 추정합니다. 같은 공구·날의 사진을 회차 순서대로 기록하면 추이와 실험용 잔여 회차를 표시합니다.')
  f=st.file_uploader('공구 날 사진을 끌어다 놓거나 선택하세요',type=['jpg','jpeg','png'])
  if f is None:
   test=df[df.split=='test'].sort_values(['tool','blade','run'])
   choice=st.selectbox('업로드할 사진이 없으면 실제 테스트 사진 선택',['선택 안 함']+test.sample_id.tolist())
   if choice!='선택 안 함':
    example=test[test.sample_id==choice].iloc[0]
    im=Image.open(ROOT/example.tool_file_name).convert('RGB')
    st.caption(f'공개 테스트 이미지 {choice} · 이력 기록 시 동일 공구와 동일 날만 사용하세요.')
  if f:
   try:im=Image.open(f).convert('RGB')
   except Exception as e:st.error(f'이미지를 읽을 수 없습니다: {e}')
 if im is not None:
  if st.button('현재 측정값을 이력에 기록',type='primary'):
   if hist and run<=hist[-1]['run']:st.error('기존 기록보다 큰 회차를 입력하세요. 같은 회차를 반복 추가하지 마세요.')
   else:hist.append({'run':float(run),'wear':predict(m,ck,im)})
if im is not None:
 wear=predict(m,ck,im);c1,c2=st.columns([1,2]);c1.image(im,caption='입력 공구 이미지',width='stretch')
 with c2:
  a,b,c=st.columns(3);a.metric('추정 마모량',f'{wear:.1f} µm')
  if len(hist)>=3:
   features=history_features([h['run'] for h in hist],[h['wear'] for h in hist]);remaining=max(0.,float(rul.predict(features[None])[0]));b.metric('잔여 관측 회차 · 실험용',f'{remaining:.1f} 회차')
  else:b.metric('잔여 관측 회차','이력 3개 필요')
  c.metric('교체 검토', '⚠ 기준 초과' if wear>=threshold else '기준 미만')
  if wear>=threshold:st.warning('설정한 마모 기준을 초과했습니다. 공구 상태를 확인하고 교체를 검토하세요.')
  else:st.caption('설정 기준 미만이라는 뜻이며, 공구 안전이나 정상 상태를 보증하지 않습니다.')
  st.caption(f"잔여 회차 모델: 이미지 이력 테스트 MAE {report['rul']['image_history_test']['MAE']:.2f}회차. 기준 모델보다 낮은 성능이며 연구용입니다.")
if hist:
 chart=pd.DataFrame(hist).set_index('run').rename(columns={'wear':'예측 마모량 (µm)'})
 chart['설정 기준 (µm)']=threshold;st.subheader('마모 추이');st.line_chart(chart,x_label='가공 회차',y_label='마모량 (µm)')
 st.download_button('이력 CSV 저장',pd.DataFrame(hist).to_csv(index=False).encode(),file_name='wear_history.csv',mime='text/csv')
with st.expander('실제 평가 결과와 한계',expanded=im is None):
 st.write('공구 분리',report['split_tools']);st.write('테스트 이미지 회귀 성능 (µm)',report['wear']['test']);st.write('평균값 기준 모델',report['mean_baseline']['test']);st.write('잔여 관측 회차 예측',report['rul'])
 st.caption('테스트 결과는 2개 공구에 한정됩니다. 현미경 → 저가형 웹캠 성능 전이는 평가하지 않았습니다. 연속 가공 이력 없이 단일 이미지로 실제 수명을 산출할 수 없습니다.')
 st.markdown('[데이터 출처 · Nonastreda / CC BY 4.0](https://data.mendeley.com/datasets/m892d2wtzh/1)')
