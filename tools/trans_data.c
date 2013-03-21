#include "trans_tools.h"

const double SUBSET1_1D_4x8[2*4]={
  1                   , 0.965300887762155946, 0.935153289354844763, 0.913283009229249432, 0.894324829952695355, 0.880063311676738458, 0.866183466827104787, 0.853497097322791554
};

const double SUBSET1_1D_8x16[2*8]={
  1                   , 0.965137275560318586, 0.934863111781725098, 0.912901489009930178, 0.893886465108923667, 0.879571607217473517, 0.86564706543863168 , 0.852921190628046166,
  0.842795373480948462, 0.831313378272043768, 0.822156688958478488, 0.813730555470264294, 0.80430299128657845 , 0.797302509841553597, 0.789267288249199228, 0.781346835940896822
};

const double SUBSET1_1D_16x32[2*16]={
  1                   , 0.964937117755906848, 0.934435546923380489, 0.912290521462185922, 0.89315710753212596 , 0.878739240191312221, 0.864715734367605515, 0.851904324586791106,
  0.841707428440539029, 0.830165202166005867, 0.82096547298106437 , 0.812501330725822646, 0.80304708820047499 , 0.796015033497699975, 0.787942730855812123, 0.779984562911727242,
  0.773867612955779549, 0.766229915808482098, 0.760141055028612511, 0.75457582447738758 , 0.747860778626181832, 0.74312707960658142 , 0.737617234005152334, 0.732172011872836648,
  0.728279143089830883, 0.722637115502609784, 0.718002552069585853, 0.713729542635013203, 0.708198399734208617, 0.704454904218173361, 0.699935549574550175, 0.695360592060045302
};

const double SUBSET3_1D_4x8[2*4]={
  1                   , 0.98005136796498693 , 0.957713574924411781, 0.941580421294460956, 0.928531346729398632, 0.917354474191168223, 0.907438969866926182, 0.898433540548949749
};

const double SUBSET3_1D_8x16[2*8]={
  1                   , 0.979985707472378897, 0.957579285072683417, 0.941393704649523211, 0.928300512015876933, 0.917086228551239557, 0.907139754104427154, 0.898108371378369075,
  0.889790946175891606, 0.882072931282638861, 0.874849677324887054, 0.868054871717973797, 0.861649709824905852, 0.855537201870884556, 0.84967501863881334 , 0.844064755536456102
};

const double SUBSET3_1D_16x32[2*16]={
  1                   , 0.979833385645487831, 0.957266023308942171, 0.940957785714492512, 0.927759903916599682, 0.916453542136722588, 0.906425658886728325, 0.897321290440588615,
  0.888936682536967893, 0.881156880166969247, 0.873875337121942874, 0.867026842393475938, 0.860573549394290538, 0.85441625899314555 , 0.848512022218074757, 0.84286226998388758 ,
  0.837433692182305123, 0.832219051301511858, 0.827217502623003198, 0.822376308784316223, 0.8176705532296209  , 0.813105070134160113, 0.808660902482214561, 0.804337882441993002,
  0.80013922861692155 , 0.796035478143399478, 0.792014997979164193, 0.788080365030657481, 0.784235634142571802, 0.780467000291831803, 0.776769073480155114, 0.773144759845687646   
};

const double SUBSET1_2D_4x8[2*4*2*4]={
  1                   , 0.968065319241571909, 0.938014058734280343, 0.917071173069408774, 0.89814814429263512 , 0.8842045154030721  , 0.870420552205966702, 0.857932048815267523,
  0.962264645146231112, 0.945737613096246488, 0.924887441183076087, 0.906852201510136458, 0.890055448569902929, 0.87685418052735864 , 0.864172289709322583, 0.852529899677461289,
  0.931802971136095337, 0.921989214782085176, 0.907733745186981467, 0.8937423330861195  , 0.879513886440549952, 0.868364397559493129, 0.85698066400626105 , 0.846198553036032308,
  0.908844137339172908, 0.901155832348194519, 0.890592908312271225, 0.879805765189072941, 0.867553645433111753, 0.858070475702575108, 0.847739482186701254, 0.837928429324908075,
  0.88973561910068788 , 0.883780168200635319, 0.875294117953762885, 0.866013997797495216, 0.855799532524259332, 0.847258432710781384, 0.838174466356914283, 0.82931233623219669 ,
  0.875045540585258075, 0.869306867100113068, 0.8624011331213981  , 0.85454362439253273 , 0.845128034825188834, 0.837951308159857633, 0.829641221663008577, 0.821416521513323339,
  0.860976449754791151, 0.855962689952442957, 0.849990577977051132, 0.843115370943231968, 0.834771268380742004, 0.82829499395784234 , 0.820673555284262002, 0.813155135435437604,
  0.848009293130841835, 0.84355039685594746 , 0.838020418278295431, 0.831964704861376858, 0.82448601186561854 , 0.818578831657175043, 0.81173287434717778 , 0.804861315164140168
};

const double SUBSET1_2D_8x16[2*8*2*8]={
  1                   , 0.967752519487544149, 0.937428833306501619, 0.91631310895981366 , 0.897264093141396102, 0.883209808929132922, 0.869329525965746086, 0.856762690427384843,
  0.846900511025535541, 0.835691080830322952, 0.826969107672746673, 0.818933002017561029, 0.809830438428076671, 0.803134017826269808, 0.795362106302003435, 0.787690705437670835,
  0.961954924291081093, 0.945274935219157619, 0.92424610169068222 , 0.906060928070372418, 0.889144721307031594, 0.875844397112008455, 0.863073894882814607, 0.851355704895122911,
  0.841621668030433367, 0.831318711641549468, 0.822831223766505082, 0.814923999068986382, 0.806517033603705191, 0.799691141952492468, 0.792198996376722975, 0.784886301462709102,
  0.931281777816421852, 0.921381192968014129, 0.906990796869103   , 0.892875067960258395, 0.878546683581894916, 0.867310516079239258, 0.855846681635714202, 0.844993924770497284,
  0.836053806428935564, 0.826089112658323188, 0.818116411621066297, 0.81074266205574641 , 0.802542206009504433, 0.796056296851092915, 0.788737985568545241, 0.781620068629307996,
  0.908141037185451805, 0.900388900847966078, 0.889720041704774256, 0.878831863926762713, 0.866494855867100067, 0.856930760887670284, 0.846526333377078832, 0.836653390428659938,
  0.82889090449344216 , 0.819467355573192613, 0.812064558629655542, 0.805202809137285702, 0.797183748563922268, 0.791223975395621393, 0.784228530812288382, 0.777298062473577134,
  0.888927836967819185, 0.882918545838175306, 0.874347650694328848, 0.864975963646774093, 0.854682212122832463, 0.846068807987367277, 0.836919023958244024, 0.827999337041125494,
  0.820535360300914562, 0.812240650858512714, 0.805279568475991914, 0.798665270121652893, 0.791417234221357568, 0.785481512556813866, 0.778858890270971771, 0.772352893134898189,
  0.874124214657913678, 0.868337238456072358, 0.861359762793239114, 0.85341874919945393 , 0.843931598712524433, 0.836686088351956125, 0.82831649234049709 , 0.820040996673757294,
  0.813576192267516984, 0.805399676755938154, 0.799061728826376605, 0.793165515563989731, 0.785977552320107353, 0.780631907233317213, 0.774135435434418206, 0.767781359517258455,
  0.859963258601435299, 0.854905626325897616, 0.848871150986304723, 0.841923434709805685, 0.83351526788595498 , 0.826977988451198409, 0.81929971436865201 , 0.811733550334082676,
  0.805777335077891843, 0.798305744827769392, 0.792300630563326669, 0.786701189287107638, 0.780029455372933334, 0.774875496820957688, 0.768762371743175743, 0.762724841050948954,
  0.84690510176443734 , 0.842403476857067512, 0.836813750587509286, 0.830697014026078207, 0.823159896692416182, 0.81720000214576527 , 0.81030404277350554 , 0.803388734710581365,
  0.797687081271963305, 0.790893941440971471, 0.785272781074388582, 0.779853863716529316, 0.773669465052654659, 0.768750720573289859, 0.763002125842184298, 0.757240585549001644,
  0.836345893740673696, 0.831851645420126795, 0.826997434494529116, 0.821805189215418341, 0.814327987222485672, 0.809239152359843228, 0.802769926061997374, 0.796283765817579647,
  0.791496195656904877, 0.784646357176256193, 0.779517456341201398, 0.77468107033105349 , 0.768377148743680127, 0.764024179898709188, 0.758367639052325404, 0.752696219195772676,
  0.824458990161440086, 0.820630334001892559, 0.815949806059630189, 0.811057718900240032, 0.804676606950521123, 0.799644958476082968, 0.793865994480091319, 0.788004642906286246,
  0.783287977925560819, 0.777246875758871081, 0.772372448650620713, 0.767661000613861555, 0.762011794691989874, 0.757594114227801096, 0.752318461206469169, 0.747055437375830134,
  0.814736842770387493, 0.810997968100112265, 0.806683200636810427, 0.802184227157318741, 0.796332827727971426, 0.791694909520326395, 0.786217744035432764, 0.780702134773094247,
  0.776525082319239779, 0.770859017247793044, 0.766381850044553836, 0.761933607211954245, 0.7564245936162175  , 0.75236599500835255 , 0.747303010613960983, 0.742165349562552534,
  0.805796863983715639, 0.802043653751527308, 0.798120597753153804, 0.793956935808122211, 0.788238421462901684, 0.784193389928592333, 0.778983314058071419, 0.773702485265075013,
  0.770055493339745101, 0.764470281448993982, 0.760383070853383969, 0.756404277031270333, 0.750966947342178393, 0.747250048093858643, 0.742247295190080369, 0.737186935778339159,
  0.79594999628604135 , 0.792720328431950949, 0.788834691213397443, 0.784765774510571168, 0.779737035722520955, 0.775718043214576158, 0.770882428525747732, 0.766141386289288406,
  0.762358712112731096, 0.757473771831270737, 0.753585881002810321, 0.749756844340881523, 0.744925053011158966, 0.741052039277713903, 0.736383828343297742, 0.731679020442787986,
  0.788520039484956858, 0.785102837146347432, 0.781478358608023504, 0.777720645813409495, 0.772613400427701658, 0.769330870792513721, 0.764533233432880821, 0.759819583869497439,
  0.756606023505698722, 0.751699714608585645, 0.748183616414752617, 0.744714328343412557, 0.739776599063227902, 0.736422748194222487, 0.731830697281605214, 0.727106089671468947,
  0.780111067648895373, 0.776956370268317764, 0.773483738354085903, 0.769870039398529715, 0.765023834463141483, 0.761924296757397923, 0.757564204180180001, 0.753101202006138837,
  0.749997039216985018, 0.745368335944091354, 0.742036880821457578, 0.738783877886318496, 0.73421668194449019 , 0.73096817123336244 , 0.726564061035920306, 0.722036172706810642,
  0.771844645469340174, 0.769001624777352011, 0.765638460575704616, 0.762113337606459496, 0.757665740378328123, 0.754565724926550607, 0.750644561116401099, 0.746431913294998917,
  0.743314500958792812, 0.739058954383143063, 0.735833103821330603, 0.732711443486120606, 0.72856383510346201 , 0.725296164823305944, 0.721149241018729015, 0.716893638026314672
};

const double SUBSET3_2D_4x8[2*4*2*4]={
  1                   , 0.981196617789956926, 0.959520334190045454, 0.943589032604141509, 0.930690125022685888, 0.919714467895986876, 0.910041022237088937, 0.901252280314526866,
  0.978811968296694701, 0.967640903577134392, 0.951457608925744713, 0.937717681241670875, 0.925987322269142044, 0.915756099867849671, 0.906602364958592921, 0.898201145383863309,
  0.955699978833517871, 0.949213507463744333, 0.938197688741848945, 0.927446727973136875, 0.917531781332417862, 0.908493978880756448, 0.900184059912716106, 0.892429224893257378,
  0.939277500263222875, 0.934396010602968952, 0.925894335780909405, 0.917182916153479466, 0.908757211736018755, 0.900807791317042361, 0.893310191881184101, 0.886189309780813672,
  0.926004333464758078, 0.9219649131872798  , 0.91487771486445757 , 0.90748845883104079 , 0.900179935385701624, 0.893135469630639167, 0.886372437762514176, 0.879860080237806597,
  0.914561222926800865, 0.911058735664653896, 0.904888809505508207, 0.898416663048911546, 0.891950764965042553, 0.885632356655233322, 0.879490521218773025, 0.873515188423576561,
  0.904345573872572972, 0.901226304945041967, 0.895708598163787784, 0.88990661439580665 , 0.884094721891169888, 0.878358888107390157, 0.872739339127630465, 0.867224393173837926,
  0.895071125012192881, 0.892255463325639653, 0.887243103267207567, 0.881956893113217122, 0.876655978860781726, 0.87139816324869257 , 0.866215693815570065, 0.861093437034028519  
}; 

const double *SUBSET1_1D[OD_NBSIZES]={
  SUBSET1_1D_4x8,
  SUBSET1_1D_8x16,
  SUBSET1_1D_16x32
};

const double *SUBSET3_1D[OD_NBSIZES]={
  SUBSET3_1D_4x8,
  SUBSET3_1D_8x16,
  SUBSET3_1D_16x32
};

const double *SUBSET1_2D[OD_NBSIZES]={
  SUBSET1_2D_4x8,
  SUBSET1_2D_8x16,
  NULL
};

const double *SUBSET3_2D[OD_NBSIZES]={
  SUBSET3_2D_4x8,
  NULL,
  NULL
};
