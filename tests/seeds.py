import pytest


@pytest.fixture
def samples():
    return (
        '''
“교육청 현장에서 다 하는 말이 현재 인력 구조에선 다른 감사를 줄이지 않는 한 사립유치원 감사에 비중을 두는 건 어렵지 않으냐고 얘기한다.”(한 교육청 관계자)

“현 인력으론 유치원을 포함해 도내 학교를 모두 감사하려면 10년이 걸린다.”(다른 교육청 관계자)

교육부가 내년도 상반기까지 대규모 유치원 등을 대상으로 종합감사를 예고한 가운데 서울시교육청만 하더라도 감사인력이 불과 4명밖에 되지 않는 등 현장 인력이 턱없이 부족한 것으로 21일 나타났다. 전국 시·도교육청 대부분이 유치원 전담 감사인력이 없는 탓에 감사인력 확대 등 보완 대책이 뒤따라야 한다는 목소리가 나온다.

■ “6명이 1100여개 유치원 감사”

이날 <한겨레>가 국회 교육위원회 소속 박용진 더불어민주당 의원을 통해 전국 17개 시·도교육청 중 13곳의 감사인력 자료를 받아 분석한 결과 평균 37.6명의 직원이 근무하고 있었다. 경기(85명)가 가장 많았고, 경남(52명), 부산(49명), 서울·전북(48명), 충남(34명), 충북(32명), 전남(31명), 대전(28명), 강원(27명), 울산(21명), 광주(20명), 세종(14명) 차례였다. 하지만 이 인원은 유치원뿐 아니라 초·중·고등학교 감사, 공무원 비위 등에 대한 사실조사를 담당하는 각 교육청 감사인력 전체를 집계한 것이다.

따라서 사립유치원 감사인력은 크게 줄어든다. 대부분의 교육청은 사립유치원 감사 담당을 따로 두지 않는다. 민원이 들어오면 감사인력 일부가 조사를 나가는 식이다. 서울시교육청의 경우 교육부가 지난 18일 ‘유치원 비리신고센터’를 꾸리기로 하자 전담팀을 만들었지만 인원이 팀장을 포함해 4명이다. 서울의 사립유치원은 650개(2018년 5월1일 기준)다. 서울시교육청 관계자는 “그동안 사립유치원 감사는 사안이 있거나 특별한 경우가 없으면 하지 못했다. 이제 급히 하려고 하니까 인력이 부족하다”며 “나쁘게 보면 졸속이라고 볼 수 있지만 일단 사립유치원 감사에 나름 최선을 다하려고 한다”고 토로했다. 경기도교육청은 감사5팀(6명)이 사립유치원 특정감사 업무를 전담해왔다. 경기도교육청 관계자는 “시민감사팀에서도 도와주지만 거긴 악성 민원도 처리해야 한다. 도내 사립유치원이 1100개가 넘는데 감사팀 6명으론 종합감사에 어려움이 있다”고 했다. 부산시교육청도 그동안 민원이 들어오면 4명이 팀을 꾸려 사실조사를 나가는 정도였다.



■ “‘배 째라’ 감사 방해도 많아”

이들이 적은 감사인력의 고충을 토로하는 이유는 사립유치원의 경우 회계서류를 제대로 갖추지 않아 들여다봐야 할 것이 많은데다, 유치원들이 ‘비협조‘로 대응하기 때문이다. 부산시교육청 관계자는 “사립유치원 조사를 하다 보면 서류가 미비되어 있거나 사람도 자주 바뀐다”며 “진행하는 과정이 너무 고되다”고 호소했다.

특히 이 관계자는 “행정처분으로 징계를 줘도 파면이 아니면 계속 유치원을 운영할 수 있고, 경고·주의 등의 경징계는 실효성이 없다는 생각이 들었다”며 “그러다 보니 유치원들이 ‘배 째라’ 태도로 감사를 방해하는 경우가 많다”고 했다. 경남도교육청 관계자도 “규정상 종합감사 주기가 3년이지만 감사인력이 적고 기관 수는 많다 보니 6년에 한번씩 돌아가고 있다. 도내 유치원부터 고등학교까지 감사기관이 1689개”라며 “우리가 1년에 할 수 있는 감사는 평균 120개 내외다. 도내 기관을 다 하려면 10년 정도 걸린다고 보면 된다”고 했다. 그는 “현장 감사의 어려움이 큰 만큼 인력 확대가 시급하다. 이는 전국 시·도가 안고 있는 문제”라고 강조했다.

박용진 의원은 “유치원을 대상으로 한 현행 특정감사에서 종합감사로 바꾸려면 감사인력 구조도 대폭 개선해야 한다”며 “제대로 된 유치원 비위 근절을 위해서는 국무총리 주관의 범정부 태스크포스(TF) 구성도 필수적”이라고 말했다. 서영지 기자 yj@hani.co.kr
        '''  # NOQA
    ,)
