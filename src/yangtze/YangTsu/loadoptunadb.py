import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
import re
import json
import logging
from datetime import datetime

# --- 配置 ---
STORAGE_URL = "sqlite:///my_optimization_history.db" # SQLite 数据库文件路径
STUDY_NAME = "xgboost_hyperparam_search_v1" # 与你的主寻优脚本中的STUDY_NAME一致
LOG_FILE_PATH_POPULATE = "populate_optuna_db_log.txt"

# --- 配置日志记录 ---
logger = logging.getLogger("populate_db")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler(LOG_FILE_PATH_POPULATE, mode='w', encoding='utf-8')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# 你的外部训练历史日志
log_data = """
[I 2025-05-18 00:45:25,251] A new study created in memory with name: no-name-45d5b364-2724-400b-9b4b-2c15d566afe3
[I 2025-05-18 00:48:13,722] Trial 0 finished with value: 0.9519780839154898 and parameters: {'n_estimators': 718, 'learning_rate': 0.007012079925836971, 'max_depth': 12, 'subsample': 0.7443934338339511, 'colsample_bytree': 0.8525985730244483, 'gamma': 0.9713678815251202, 'lambda': 0.2785976772582166, 'alpha': 1.1726802977165098e-08}. Best is trial 0 with value: 0.9519780839154898.
[I 2025-05-18 00:58:28,209] Trial 1 finished with value: 0.9488917677988358 and parameters: {'n_estimators': 2035, 'learning_rate': 0.001656057871843877, 'max_depth': 13, 'subsample': 0.7948707721490318, 'colsample_bytree': 0.9031624019345381, 'gamma': 0.46754170910914106, 'lambda': 0.217478680370962, 'alpha': 5.951911498230732}. Best is trial 0 with value: 0.9519780839154898.
[I 2025-05-18 00:59:39,938] Trial 2 finished with value: 0.9818246828286973 and parameters: {'n_estimators': 718, 'learning_rate': 0.27355203968828584, 'max_depth': 13, 'subsample': 0.8821287099851913, 'colsample_bytree': 0.6359762085388919, 'gamma': 0.7963653900622107, 'lambda': 3.7397726925165233e-07, 'alpha': 9.674960248200315e-05}. Best is trial 2 with value: 0.9818246828286973.
[I 2025-05-18 01:04:36,275] Trial 4 finished with value: 0.976023765159064 and parameters: {'n_estimators': 1369, 'learning_rate': 0.4683799100760631, 'max_depth': 10, 'subsample': 0.6115044926819482, 'colsample_bytree': 0.86473010722483, 'gamma': 0.6825997504261366, 'lambda': 5.411953884562014e-07, 'alpha': 1.908931930406256e-08}. Best is trial 2 with value: 0.9818246828286973.
[I 2025-05-18 01:07:21,113] Trial 5 finished with value: 0.9268188718676637 and parameters: {'n_estimators': 1976, 'learning_rate': 0.0021726562297615204, 'max_depth': 6, 'subsample': 0.5513319698606656, 'colsample_bytree': 0.9016926055512742, 'gamma': 0.42873901570036166, 'lambda': 1.958742558625567e-06, 'alpha': 0.0014698109993754756}. Best is trial 2 with value: 0.9818246828286973.
[I 2025-05-18 01:15:24,060] Trial 6 finished with value: 0.9614927845774889 and parameters: {'n_estimators': 2366, 'learning_rate': 0.004174049695441833, 'max_depth': 12, 'subsample': 0.8757666251420504, 'colsample_bytree': 0.680003990923393, 'gamma': 0.10974431620689629, 'lambda': 1.234422889727366e-07, 'alpha': 1.0189456751150507e-06}. Best is trial 2 with value: 0.9818246828286973.
[I 2025-05-18 01:22:48,758] Trial 7 finished with value: 0.9676414393882465 and parameters: {'n_estimators': 1154, 'learning_rate': 0.00698877845261341, 'max_depth': 14, 'subsample': 0.7791386657395465, 'colsample_bytree': 0.5415142097122159, 'gamma': 0.13546079954929335, 'lambda': 1.9519092487552014e-05, 'alpha': 0.2661818170838475}. Best is trial 2 with value: 0.9818246828286973.
[I 2025-05-18 01:27:23,114] Trial 8 finished with value: 0.9591231780755392 and parameters: {'n_estimators': 774, 'learning_rate': 0.007482349914381423, 'max_depth': 13, 'subsample': 0.8027909578436907, 'colsample_bytree': 0.8805932158728492, 'gamma': 0.10812224619120758, 'lambda': 2.319012546392842e-07, 'alpha': 7.325602125176082e-07}. Best is trial 2 with value: 0.9818246828286973.
[I 2025-05-18 01:30:46,466] Trial 9 finished with value: 0.983136935929579 and parameters: {'n_estimators': 2044, 'learning_rate': 0.3946720626285836, 'max_depth': 14, 'subsample': 0.8555637291044709, 'colsample_bytree': 0.6145647723034193, 'gamma': 0.3095463631756792, 'lambda': 0.07153008639139238, 'alpha': 5.308893710688652}. Best is trial 9 with value: 0.983136935929579.
[I 2025-05-18 01:32:44,862] Trial 10 finished with value: 0.9352929878854478 and parameters: {'n_estimators': 2494, 'learning_rate': 0.04392582935458374, 'max_depth': 3, 'subsample': 0.9910150362882679, 'colsample_bytree': 0.5178795330317768, 'gamma': 0.3003155853310795, 'lambda': 0.0012757222867200181, 'alpha': 0.05183073698893637}. Best is trial 9 with value: 0.983136935929579.
[I 2025-05-18 01:33:41,445] Trial 11 finished with value: 0.9798199957207181 and parameters: {'n_estimators': 1616, 'learning_rate': 0.4726405719755079, 'max_depth': 15, 'subsample': 0.9395323961864944, 'colsample_bytree': 0.674339760209314, 'gamma': 0.6592906252823639, 'lambda': 0.0010011983664979488, 'alpha': 0.0007815311115024722}. Best is trial 9 with value: 0.983136935929579.
[I 2025-05-18 01:35:54,246] Trial 12 finished with value: 0.982068896357805 and parameters: {'n_estimators': 1070, 'learning_rate': 0.13544462747895863, 'max_depth': 11, 'subsample': 0.8841228073908317, 'colsample_bytree': 0.6245627741386445, 'gamma': 0.3080334564940874, 'lambda': 1.208078649820873e-08, 'alpha': 0.019202886076042494}. Best is trial 9 with value: 0.983136935929579.
[I 2025-05-18 01:37:59,742] Trial 13 finished with value: 0.9770416398266238 and parameters: {'n_estimators': 1129, 'learning_rate': 0.11669301907304759, 'max_depth': 10, 'subsample': 0.8742173184374223, 'colsample_bytree': 0.6022414345448315, 'gamma': 0.30040077868731196, 'lambda': 0.01690719076826415, 'alpha': 9.918804538282368}. Best is trial 9 with value: 0.983136935929579.
[I 2025-05-18 01:39:28,003] Trial 14 finished with value: 0.9716895681483267 and parameters: {'n_estimators': 1069, 'learning_rate': 0.1384225490354471, 'max_depth': 8, 'subsample': 0.6992598673115924, 'colsample_bytree': 0.7672854866727796, 'gamma': 0.3200783788908013, 'lambda': 1.1528056201587347e-08, 'alpha': 0.044910842233665045}. Best is trial 9 with value: 0.983136935929579.
[I 2025-05-18 01:48:02,637] Trial 15 finished with value: 0.9856468531176371 and parameters: {'n_estimators': 1567, 'learning_rate': 0.03190108086337776, 'max_depth': 15, 'subsample': 0.9308058747736159, 'colsample_bytree': 0.7517500226375462, 'gamma': 0.23710286659023017, 'lambda': 4.8645995490385064e-05, 'alpha': 0.6400568254751905}. Best is trial 15 with value: 0.9856468531176371.
[I 2025-05-18 01:58:10,965] Trial 16 finished with value: 0.9841986455051804 and parameters: {'n_estimators': 1693, 'learning_rate': 0.02331527704739685, 'max_depth': 15, 'subsample': 0.9859876592992003, 'colsample_bytree': 0.7614659157203131, 'gamma': 0.0031905338786306636, 'lambda': 5.6015157993678665e-05, 'alpha': 0.7219421386865369}. Best is trial 15 with value: 0.9856468531176371.
[I 2025-05-18 02:07:56,245] Trial 17 finished with value: 0.9836339684436517 and parameters: {'n_estimators': 1614, 'learning_rate': 0.02112124903325072, 'max_depth': 15, 'subsample': 0.9566021751460724, 'colsample_bytree': 0.7740919505111992, 'gamma': 0.016539489367668813, 'lambda': 3.4138972236405394e-05, 'alpha': 0.4133996571234476}. Best is trial 15 with value: 0.9856468531176371.
[I 2025-05-18 02:09:45,857] Trial 18 finished with value: 0.9487472267840599 and parameters: {'n_estimators': 1447, 'learning_rate': 0.023165700493995932, 'max_depth': 7, 'subsample': 0.9957156154380726, 'colsample_bytree': 0.8046855035515929, 'gamma': 0.006686800008247451, 'lambda': 7.209623537546983e-05, 'alpha': 0.0061384370826199726}. Best is trial 15 with value: 0.9856468531176371.
[I 2025-05-18 02:11:25,577] Trial 19 finished with value: 0.9349356596216689 and parameters: {'n_estimators': 1782, 'learning_rate': 0.024442893241835142, 'max_depth': 4, 'subsample': 0.9321511480467881, 'colsample_bytree': 0.7094856194760204, 'gamma': 0.19063076932811246, 'lambda': 8.253569906977359e-06, 'alpha': 0.39425787270286206}. Best is trial 15 with value: 0.9856468531176371.
[I 2025-05-18 02:19:02,576] Trial 20 finished with value: 0.9860914182020476 and parameters: {'n_estimators': 1391, 'learning_rate': 0.05288611610019165, 'max_depth': 15, 'subsample': 0.7048686035069371, 'colsample_bytree': 0.8152532666208882, 'gamma': 0.20427993318953397, 'lambda': 0.0002010902827279487, 'alpha': 1.1209918586545113}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 02:24:57,272] Trial 21 finished with value: 0.9846273776516076 and parameters: {'n_estimators': 1315, 'learning_rate': 0.04580016999955667, 'max_depth': 14, 'subsample': 0.6800448432465434, 'colsample_bytree': 0.8182550095150586, 'gamma': 0.20802883156697297, 'lambda': 0.00027077130165657174, 'alpha': 0.5461679939648083}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 02:32:07,316] Trial 22 finished with value: 0.9856813148481677 and parameters: {'n_estimators': 1293, 'learning_rate': 0.053752348597253335, 'max_depth': 15, 'subsample': 0.6746370644966275, 'colsample_bytree': 0.8225993089018886, 'gamma': 0.19388835193825055, 'lambda': 0.0004832635781178572, 'alpha': 1.300423595044934}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 02:38:23,449] Trial 23 finished with value: 0.9854341991445774 and parameters: {'n_estimators': 1266, 'learning_rate': 0.05768549236205113, 'max_depth': 15, 'subsample': 0.6699334438503222, 'colsample_bytree': 0.9632873090106544, 'gamma': 0.5350723838007486, 'lambda': 0.005071928654577362, 'alpha': 2.1163026350546246}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 02:41:28,999] Trial 24 finished with value: 0.9642811992558139 and parameters: {'n_estimators': 915, 'learning_rate': 0.013266233398539846, 'max_depth': 12, 'subsample': 0.6258619534013005, 'colsample_bytree': 0.7208355452479855, 'gamma': 0.4222612081912654, 'lambda': 0.00029706615557860064, 'alpha': 0.004379104927321376}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 02:47:29,316] Trial 25 finished with value: 0.9860722893276616 and parameters: {'n_estimators': 1481, 'learning_rate': 0.06548055296297502, 'max_depth': 14, 'subsample': 0.7328088499312534, 'colsample_bytree': 0.8329882204591967, 'gamma': 0.21885783724751395, 'lambda': 0.004036940080919897, 'alpha': 0.07311784828670682}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 02:51:21,633] Trial 26 finished with value: 0.9845314446475769 and parameters: {'n_estimators': 895, 'learning_rate': 0.07120285368271209, 'max_depth': 14, 'subsample': 0.7299778751623316, 'colsample_bytree': 0.8114789595842812, 'gamma': 0.14574591018771493, 'lambda': 0.004949278413659157, 'alpha': 0.07695795329644642}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 02:54:28,190] Trial 27 finished with value: 0.9816839263436404 and parameters: {'n_estimators': 1430, 'learning_rate': 0.17084064811619634, 'max_depth': 11, 'subsample': 0.5196710398085865, 'colsample_bytree': 0.9346272283422461, 'gamma': 0.5423420566198147, 'lambda': 0.029922818999577102, 'alpha': 0.008527191862821545}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 02:57:31,472] Trial 28 finished with value: 0.9841475053366557 and parameters: {'n_estimators': 1281, 'learning_rate': 0.23418138708969846, 'max_depth': 13, 'subsample': 0.7095635975646728, 'colsample_bytree': 0.8295701745794688, 'gamma': 0.36535548777676663, 'lambda': 0.0012716340183634618, 'alpha': 0.07829728491799169}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 02:59:34,313] Trial 29 finished with value: 0.9778370371097015 and parameters: {'n_estimators': 947, 'learning_rate': 0.08237361310533693, 'max_depth': 11, 'subsample': 0.7541035337762033, 'colsample_bytree': 0.842339851937068, 'gamma': 0.977187511117946, 'lambda': 1.9352716599747124, 'alpha': 3.442445080795034}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 03:01:48,234] Trial 30 finished with value: 0.9589033422773919 and parameters: {'n_estimators': 578, 'learning_rate': 0.013977473378430536, 'max_depth': 12, 'subsample': 0.6400826720324237, 'colsample_bytree': 0.7903886604546726, 'gamma': 0.22897108246226475, 'lambda': 4.228778131195981e-06, 'alpha': 0.1353802260047189}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 03:09:45,131] Trial 31 finished with value: 0.9853664348720345 and parameters: {'n_estimators': 1540, 'learning_rate': 0.0372859871393372, 'max_depth': 15, 'subsample': 0.8306917162244325, 'colsample_bytree': 0.7183540029220103, 'gamma': 0.22889552867299376, 'lambda': 0.00014244804467837782, 'alpha': 1.7152189473728123}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 03:16:43,488] Trial 32 finished with value: 0.9839293588735267 and parameters: {'n_estimators': 1521, 'learning_rate': 0.03538846269068019, 'max_depth': 14, 'subsample': 0.7522031272629728, 'colsample_bytree': 0.7447900016549424, 'gamma': 0.07517427559839324, 'lambda': 0.003418412249941654, 'alpha': 1.4785493400943501}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 03:20:47,262] Trial 33 finished with value: 0.9793296333743371 and parameters: {'n_estimators': 1216, 'learning_rate': 0.056590185360466234, 'max_depth': 13, 'subsample': 0.5767362722494374, 'colsample_bytree': 0.8690682685598177, 'gamma': 0.18059901562492775, 'lambda': 0.0005533923636171783, 'alpha': 9.76071186433849}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 03:31:48,837] Trial 34 finished with value: 0.9819599759820545 and parameters: {'n_estimators': 1824, 'learning_rate': 0.01433118092150991, 'max_depth': 15, 'subsample': 0.6640551984685292, 'colsample_bytree': 0.9097218304533068, 'gamma': 0.26152500490813685, 'lambda': 0.3816171037721248, 'alpha': 6.364788333992061e-05}. Best is trial 20 with value: 0.9860914182020476.
[I 2025-05-18 03:38:45,013] Trial 35 finished with value: 0.9867239343080104 and parameters: {'n_estimators': 1961, 'learning_rate': 0.08148819575292222, 'max_depth': 14, 'subsample': 0.7211670065391603, 'colsample_bytree': 0.84674190405471, 'gamma': 0.3677495786830702, 'lambda': 0.012339402904032905, 'alpha': 1.1481405582782804}. Best is trial 35 with value: 0.9867239343080104.
[I 2025-05-18 03:44:52,583] Trial 36 finished with value: 0.986220323456852 and parameters: {'n_estimators': 2163, 'learning_rate': 0.08656757013193256, 'max_depth': 13, 'subsample': 0.709837912631171, 'colsample_bytree': 0.8559118625356164, 'gamma': 0.3852395829770314, 'lambda': 0.23715037291485583, 'alpha': 0.022705744833505775}. Best is trial 35 with value: 0.9867239343080104.
[I 2025-05-18 03:48:47,338] Trial 37 finished with value: 0.9850671695160074 and parameters: {'n_estimators': 2266, 'learning_rate': 0.20665753838493744, 'max_depth': 12, 'subsample': 0.7289066209821345, 'colsample_bytree': 0.9425782459841792, 'gamma': 0.38481037544123997, 'lambda': 0.9126959327257776, 'alpha': 0.025951867068802866}. Best is trial 35 with value: 0.9867239343080104.
[I 2025-05-18 03:53:46,030] Trial 38 finished with value: 0.9862428298698291 and parameters: {'n_estimators': 2115, 'learning_rate': 0.0950258130502709, 'max_depth': 13, 'subsample': 0.7852822206946048, 'colsample_bytree': 0.8924682169629599, 'gamma': 0.4818544711608178, 'lambda': 0.10495809455620367, 'alpha': 0.0004739461129161449}. Best is trial 35 with value: 0.9867239343080104.
[I 2025-05-18 03:57:25,971] Trial 39 finished with value: 0.9827956547194193 and parameters: {'n_estimators': 2207, 'learning_rate': 0.09523154861061227, 'max_depth': 10, 'subsample': 0.7845799465439514, 'colsample_bytree': 0.994149398346107, 'gamma': 0.6323060802410889, 'lambda': 0.13712621329404526, 'alpha': 0.00026827114063725986}. Best is trial 35 with value: 0.9867239343080104.
[I 2025-05-18 04:02:41,702] Trial 40 finished with value: 0.9856678434997315 and parameters: {'n_estimators': 1992, 'learning_rate': 0.10326200254541172, 'max_depth': 13, 'subsample': 0.8092273362671708, 'colsample_bytree': 0.8996196749465734, 'gamma': 0.4792249358292987, 'lambda': 7.138925566158392, 'alpha': 3.772474346200693e-06}. Best is trial 35 with value: 0.9867239343080104.
[I 2025-05-18 04:09:37,679] Trial 41 finished with value: 0.9868991283507776 and parameters: {'n_estimators': 2104, 'learning_rate': 0.0704168538408898, 'max_depth': 14, 'subsample': 0.7606516160412129, 'colsample_bytree': 0.8589145057140283, 'gamma': 0.3814671703583167, 'lambda': 0.02826908505045723, 'alpha': 0.16220644839732679}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 04:11:57,218] Trial 42 finished with value: 0.9826218448846854 and parameters: {'n_estimators': 2109, 'learning_rate': 0.27449481974989043, 'max_depth': 13, 'subsample': 0.6968408096822988, 'colsample_bytree': 0.8530067856476531, 'gamma': 0.5916609354152665, 'lambda': 0.02924180680229433, 'alpha': 0.0019942116671117335}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 04:17:24,623] Trial 43 finished with value: 0.9865694551176777 and parameters: {'n_estimators': 2144, 'learning_rate': 0.08909198902872933, 'max_depth': 14, 'subsample': 0.8261164293585093, 'colsample_bytree': 0.8832696160733673, 'gamma': 0.4335628568401465, 'lambda': 0.39444690341986166, 'alpha': 0.2048367468137038}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 04:21:04,130] Trial 44 finished with value: 0.9857744092865167 and parameters: {'n_estimators': 1911, 'learning_rate': 0.16660654541057654, 'max_depth': 14, 'subsample': 0.7716073024965986, 'colsample_bytree': 0.8845720480405255, 'gamma': 0.4165092533433429, 'lambda': 0.3760959567786662, 'alpha': 0.20293470960787363}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 04:23:09,772] Trial 45 finished with value: 0.9832146891186533 and parameters: {'n_estimators': 2386, 'learning_rate': 0.31178344595632135, 'max_depth': 12, 'subsample': 0.8265569887385398, 'colsample_bytree': 0.9054294837113531, 'gamma': 0.7665343936999394, 'lambda': 1.865352359853347, 'alpha': 0.017859497738657273}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 04:37:03,257] Trial 46 finished with value: 0.9503591413509217 and parameters: {'n_estimators': 2139, 'learning_rate': 0.0012915449363217432, 'max_depth': 13, 'subsample': 0.7685465147226331, 'colsample_bytree': 0.9420714712486465, 'gamma': 0.4572720368238271, 'lambda': 0.11927577849562578, 'alpha': 0.00039275121638362005}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 04:41:17,598] Trial 47 finished with value: 0.9854088998683661 and parameters: {'n_estimators': 2331, 'learning_rate': 0.13220312965308356, 'max_depth': 11, 'subsample': 0.8432117262905029, 'colsample_bytree': 0.9247021700127378, 'gamma': 0.36598021719575496, 'lambda': 0.052982973444946395, 'alpha': 8.024069797870954e-06}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 04:44:14,137] Trial 48 finished with value: 0.9787891432325024 and parameters: {'n_estimators': 2086, 'learning_rate': 0.08501156343319827, 'max_depth': 9, 'subsample': 0.8015827667311771, 'colsample_bytree': 0.8561183973814745, 'gamma': 0.5197270329058616, 'lambda': 0.011572571287486367, 'alpha': 1.1267210050503135e-07}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 04:46:24,660] Trial 49 finished with value: 0.98452380674593 and parameters: {'n_estimators': 2495, 'learning_rate': 0.1867270151929091, 'max_depth': 14, 'subsample': 0.9079888536892778, 'colsample_bytree': 0.9711717752076867, 'gamma': 0.5719712404949467, 'lambda': 1.5601257890507785, 'alpha': 0.0017768938256628802}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 04:50:26,998] Trial 50 finished with value: 0.9856794504304878 and parameters: {'n_estimators': 1900, 'learning_rate': 0.1205261816652673, 'max_depth': 12, 'subsample': 0.8184201080448814, 'colsample_bytree': 0.8814585601758557, 'gamma': 0.4586789221420917, 'lambda': 0.24137440737294402, 'alpha': 0.15201217466359354}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 04:57:35,853] Trial 51 finished with value: 0.9867423992926169 and parameters: {'n_estimators': 2196, 'learning_rate': 0.07578730125314163, 'max_depth': 14, 'subsample': 0.7112156295688871, 'colsample_bytree': 0.8646857768472801, 'gamma': 0.3763606250884941, 'lambda': 0.06371131994591023, 'alpha': 0.25475321181012733}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 05:04:59,796] Trial 52 finished with value: 0.9863840976281957 and parameters: {'n_estimators': 2206, 'learning_rate': 0.08634680129413391, 'max_depth': 14, 'subsample': 0.6474588490673616, 'colsample_bytree': 0.8704765055899375, 'gamma': 0.34495532487747327, 'lambda': 0.594811268962626, 'alpha': 0.009384325372262267}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 05:13:30,889] Trial 53 finished with value: 0.9861757862939499 and parameters: {'n_estimators': 2287, 'learning_rate': 0.06995107815762094, 'max_depth': 14, 'subsample': 0.5988497395296642, 'colsample_bytree': 0.8845583442049176, 'gamma': 0.33593111048982316, 'lambda': 0.6956341821747972, 'alpha': 0.009988985067124166}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 05:15:25,180] Trial 54 finished with value: 0.943769175787887 and parameters: {'n_estimators': 2018, 'learning_rate': 0.03052350874874676, 'max_depth': 5, 'subsample': 0.6531253660289966, 'colsample_bytree': 0.9180044359676799, 'gamma': 0.2763611717137594, 'lambda': 0.09111541060201951, 'alpha': 0.00015207813475690994}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 05:19:17,282] Trial 55 finished with value: 0.9862042934127546 and parameters: {'n_estimators': 2395, 'learning_rate': 0.14862965144603973, 'max_depth': 14, 'subsample': 0.7889378695675593, 'colsample_bytree': 0.8694564195056887, 'gamma': 0.34129554133299794, 'lambda': 0.020504260523071492, 'alpha': 0.0008019871948761676}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 05:30:23,514] Trial 56 finished with value: 0.9632384055004022 and parameters: {'n_estimators': 2202, 'learning_rate': 0.003533653058137867, 'max_depth': 13, 'subsample': 0.850113229748425, 'colsample_bytree': 0.7802834985530666, 'gamma': 0.4930323942683901, 'lambda': 0.010112750040283777, 'alpha': 0.0035531487476770955}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 05:37:29,835] Trial 57 finished with value: 0.985787731208478 and parameters: {'n_estimators': 1731, 'learning_rate': 0.043582056645884, 'max_depth': 14, 'subsample': 0.7621513509548348, 'colsample_bytree': 0.8982737703341084, 'gamma': 0.42010422851289553, 'lambda': 0.06486506907488991, 'alpha': 0.2768412968242799}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 05:44:39,140] Trial 58 finished with value: 0.9854896009022058 and parameters: {'n_estimators': 1922, 'learning_rate': 0.11228549224562959, 'max_depth': 15, 'subsample': 0.7419443991633283, 'colsample_bytree': 0.584255260741197, 'gamma': 0.2825738702560525, 'lambda': 4.716263053864094, 'alpha': 3.237738388632324}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 05:47:46,692] Trial 59 finished with value: 0.9823555598357774 and parameters: {'n_estimators': 2049, 'learning_rate': 0.33857403565644223, 'max_depth': 13, 'subsample': 0.6393417788029663, 'colsample_bytree': 0.8392575973452983, 'gamma': 0.40124849363579507, 'lambda': 0.711204392693664, 'alpha': 0.730132008642274}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 05:53:10,633] Trial 60 finished with value: 0.9865354265831559 and parameters: {'n_estimators': 2435, 'learning_rate': 0.06879248161224488, 'max_depth': 15, 'subsample': 0.8629566512294817, 'colsample_bytree': 0.7977747173265597, 'gamma': 0.4477273884187639, 'lambda': 0.04542380771897262, 'alpha': 0.04309404963033434}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 05:58:13,429] Trial 61 finished with value: 0.9864817604608809 and parameters: {'n_estimators': 2420, 'learning_rate': 0.07917304039028436, 'max_depth': 15, 'subsample': 0.8676290829026923, 'colsample_bytree': 0.7965855789967763, 'gamma': 0.43312875120917216, 'lambda': 0.0545155170664313, 'alpha': 0.03430342832669055}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 06:03:33,518] Trial 62 finished with value: 0.9864625029536402 and parameters: {'n_estimators': 2442, 'learning_rate': 0.0700907696055756, 'max_depth': 15, 'subsample': 0.8686926170629071, 'colsample_bytree': 0.789996735617481, 'gamma': 0.4508382777239839, 'lambda': 0.008267752898160551, 'alpha': 0.04050091728780911}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 06:10:42,326] Trial 63 finished with value: 0.9863303724418219 and parameters: {'n_estimators': 2427, 'learning_rate': 0.04616951093813746, 'max_depth': 15, 'subsample': 0.8953437549027068, 'colsample_bytree': 0.786340861075336, 'gamma': 0.5143675716330861, 'lambda': 0.010064228681701867, 'alpha': 0.03290534015877739}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 06:16:19,468] Trial 64 finished with value: 0.9865128880469679 and parameters: {'n_estimators': 2304, 'learning_rate': 0.06552442428047707, 'max_depth': 15, 'subsample': 0.8634578723014344, 'colsample_bytree': 0.8047851714376449, 'gamma': 0.44329427019129913, 'lambda': 0.03858736333847899, 'alpha': 0.11383076808396017}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 06:22:21,160] Trial 65 finished with value: 0.9864921752776795 and parameters: {'n_estimators': 2299, 'learning_rate': 0.0634708230465487, 'max_depth': 15, 'subsample': 0.865703036086721, 'colsample_bytree': 0.738244160717777, 'gamma': 0.44599315148434554, 'lambda': 0.05158322955023464, 'alpha': 0.15964565357556346}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 06:28:49,082] Trial 66 finished with value: 0.9821569427351894 and parameters: {'n_estimators': 2285, 'learning_rate': 0.026897343007744975, 'max_depth': 14, 'subsample': 0.9161272355498197, 'colsample_bytree': 0.6730892820777106, 'gamma': 0.9144981262672452, 'lambda': 0.03575587308701628, 'alpha': 0.11127060576724915}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 06:34:08,626] Trial 67 finished with value: 0.9859677447337502 and parameters: {'n_estimators': 2332, 'learning_rate': 0.05858596172722831, 'max_depth': 15, 'subsample': 0.8885256446280885, 'colsample_bytree': 0.8241007203654488, 'gamma': 0.5443761324344004, 'lambda': 0.019784027618300434, 'alpha': 0.296099643686057}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 06:43:09,951] Trial 68 finished with value: 0.9820508640980871 and parameters: {'n_estimators': 2215, 'learning_rate': 0.018393442300420716, 'max_depth': 14, 'subsample': 0.9652963818846587, 'colsample_bytree': 0.7400374350218889, 'gamma': 0.580048205309525, 'lambda': 0.0021950253563051904, 'alpha': 0.8357557236076552}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 06:45:43,721] Trial 69 finished with value: 0.9634808567162632 and parameters: {'n_estimators': 2313, 'learning_rate': 0.046691558057740265, 'max_depth': 7, 'subsample': 0.8377067086307813, 'colsample_bytree': 0.762038258927524, 'gamma': 0.3796662125361834, 'lambda': 0.17659342576439643, 'alpha': 0.4679618023913234}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 06:54:12,138] Trial 70 finished with value: 0.9852300411704388 and parameters: {'n_estimators': 1955, 'learning_rate': 0.03828975649020561, 'max_depth': 15, 'subsample': 0.8535685144994577, 'colsample_bytree': 0.8058419114955419, 'gamma': 0.4398960905058707, 'lambda': 6.504896024597212e-08, 'alpha': 4.053254039115158}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 06:58:59,772] Trial 71 finished with value: 0.9862916946581349 and parameters: {'n_estimators': 2439, 'learning_rate': 0.07530470476489713, 'max_depth': 15, 'subsample': 0.862305790951109, 'colsample_bytree': 0.700806636705095, 'gamma': 0.43024523875809884, 'lambda': 0.05316186329964265, 'alpha': 0.08222386400474643}. Best is trial 41 with value: 0.9868991283507776.
[I 2025-05-18 07:06:21,336] Trial 72 finished with value: 0.9870821735461613 and parameters: {'n_estimators': 2360, 'learning_rate': 0.06129925300575542, 'max_depth': 14, 'subsample': 0.8808035408597943, 'colsample_bytree': 0.7343224953487896, 'gamma': 0.3057738154991831, 'lambda': 0.039938266045508757, 'alpha': 0.19727209733436937}. Best is trial 72 with value: 0.9870821735461613.
[I 2025-05-18 07:13:13,044] Trial 73 finished with value: 0.9869526252595275 and parameters: {'n_estimators': 2249, 'learning_rate': 0.063149852981033, 'max_depth': 14, 'subsample': 0.8969793012334497, 'colsample_bytree': 0.732335542550544, 'gamma': 0.3195092460729692, 'lambda': 0.02773988561500493, 'alpha': 0.2111960186576006}. Best is trial 72 with value: 0.9870821735461613.
[I 2025-05-18 07:17:19,097] Trial 74 finished with value: 0.9864262240744941 and parameters: {'n_estimators': 2251, 'learning_rate': 0.11081027597045391, 'max_depth': 14, 'subsample': 0.9091516211533601, 'colsample_bytree': 0.6572134563089824, 'gamma': 0.29928163197967167, 'lambda': 0.0031843732622556732, 'alpha': 0.3651901770049153}. Best is trial 72 with value: 0.9870821735461613.
[I 2025-05-18 07:25:17,537] Trial 75 finished with value: 0.9857242204630171 and parameters: {'n_estimators': 2069, 'learning_rate': 0.0584038725028168, 'max_depth': 14, 'subsample': 0.6896471689593658, 'colsample_bytree': 0.70034594358523, 'gamma': 0.35010698127108414, 'lambda': 0.017443013472179948, 'alpha': 2.4684734798644583}. Best is trial 72 with value: 0.9870821735461613.
[I 2025-05-18 07:28:00,003] Trial 76 finished with value: 0.9694240180329501 and parameters: {'n_estimators': 2181, 'learning_rate': 0.05046200476801789, 'max_depth': 8, 'subsample': 0.947770584634526, 'colsample_bytree': 0.7271316631282245, 'gamma': 0.24622413799638548, 'lambda': 0.006584304554285708, 'alpha': 1.1356133751777437}. Best is trial 72 with value: 0.9870821735461613.
[I 2025-05-18 07:37:11,563] Trial 77 finished with value: 0.9863076743320993 and parameters: {'n_estimators': 2346, 'learning_rate': 0.03971902566615107, 'max_depth': 14, 'subsample': 0.7241619726563794, 'colsample_bytree': 0.8387658704245442, 'gamma': 0.3117775166352686, 'lambda': 0.0019069424959695742, 'alpha': 0.05806951352887347}. Best is trial 72 with value: 0.9870821735461613.
[I 2025-05-18 07:44:13,011] Trial 78 finished with value: 0.9839620928358443 and parameters: {'n_estimators': 2247, 'learning_rate': 0.031015105062451538, 'max_depth': 13, 'subsample': 0.9240332463861916, 'colsample_bytree': 0.7689816797623702, 'gamma': 0.3935338077793531, 'lambda': 0.18603981364333774, 'alpha': 0.17119131406275984}. Best is trial 72 with value: 0.9870821735461613.
[I 2025-05-18 07:49:32,328] Trial 79 finished with value: 0.9857107917405565 and parameters: {'n_estimators': 2159, 'learning_rate': 0.133804087184111, 'max_depth': 14, 'subsample': 0.897915692615578, 'colsample_bytree': 0.8047883594177991, 'gamma': 0.3264442593015109, 'lambda': 0.028309068724896597, 'alpha': 6.657917766911489}. Best is trial 72 with value: 0.9870821735461613.
[I 2025-05-18 07:51:22,491] Trial 80 finished with value: 0.941219856074277 and parameters: {'n_estimators': 2472, 'learning_rate': 0.10104114255582479, 'max_depth': 3, 'subsample': 0.8805099149931838, 'colsample_bytree': 0.7513645650494196, 'gamma': 0.4050328883261925, 'lambda': 0.09816899851873297, 'alpha': 0.590403702955011}. Best is trial 72 with value: 0.9870821735461613.
[I 2025-05-18 07:58:56,316] Trial 81 finished with value: 0.9871777819748337 and parameters: {'n_estimators': 2365, 'learning_rate': 0.06484530828688133, 'max_depth': 15, 'subsample': 0.8193131469026914, 'colsample_bytree': 0.7382062699145382, 'gamma': 0.36607924388692653, 'lambda': 0.36731062425223354, 'alpha': 0.22413598365642257}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 08:06:47,291] Trial 82 finished with value: 0.9871619961923918 and parameters: {'n_estimators': 2366, 'learning_rate': 0.062409305349043674, 'max_depth': 15, 'subsample': 0.8060741238338825, 'colsample_bytree': 0.7304286033599195, 'gamma': 0.3668588654272414, 'lambda': 0.3592184801933573, 'alpha': 0.2602286199043307}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 08:11:54,788] Trial 83 finished with value: 0.9865944085348696 and parameters: {'n_estimators': 2375, 'learning_rate': 0.08846469811529402, 'max_depth': 15, 'subsample': 0.8138602018798915, 'colsample_bytree': 0.6890906363018433, 'gamma': 0.3568722150636854, 'lambda': 0.2690708541872698, 'alpha': 0.2623409614488584}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 08:16:35,119] Trial 84 finished with value: 0.9860149024547529 and parameters: {'n_estimators': 2379, 'learning_rate': 0.15156280938080421, 'max_depth': 14, 'subsample': 0.815247516756876, 'colsample_bytree': 0.6882191703667089, 'gamma': 0.36562155232629684, 'lambda': 0.334934251388936, 'alpha': 1.724065603333951}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 08:22:32,107] Trial 85 finished with value: 0.9867508906143925 and parameters: {'n_estimators': 2122, 'learning_rate': 0.09851175979550397, 'max_depth': 13, 'subsample': 0.7995862597343153, 'colsample_bytree': 0.7301081348145928, 'gamma': 0.2695671269906868, 'lambda': 0.3720546968311421, 'alpha': 0.252990744927445}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 08:29:38,060] Trial 86 finished with value: 0.9850007414596706 and parameters: {'n_estimators': 2357, 'learning_rate': 0.052640332329345126, 'max_depth': 13, 'subsample': 0.7159835383926354, 'colsample_bytree': 0.6515126390102405, 'gamma': 0.2586658442245525, 'lambda': 1.224146965761701, 'alpha': 0.0152991723934936}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 08:32:26,650] Trial 87 finished with value: 0.9853250888392555 and parameters: {'n_estimators': 1852, 'learning_rate': 0.22868307188872694, 'max_depth': 15, 'subsample': 0.7928813618429722, 'colsample_bytree': 0.7302438800404568, 'gamma': 0.28877478054592326, 'lambda': 0.16318880978550343, 'alpha': 0.3691891908375184}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 08:38:02,898] Trial 88 finished with value: 0.9853110954252612 and parameters: {'n_estimators': 2240, 'learning_rate': 0.123350010915782, 'max_depth': 12, 'subsample': 0.7365709790831979, 'colsample_bytree': 0.7183800389927706, 'gamma': 0.31510189981606296, 'lambda': 3.748814109511486, 'alpha': 1.1214759646867503}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 08:46:07,850] Trial 89 finished with value: 0.9720448335222522 and parameters: {'n_estimators': 2129, 'learning_rate': 0.008953237296373618, 'max_depth': 13, 'subsample': 0.8018906690213159, 'colsample_bytree': 0.70385126700732, 'gamma': 0.17439528286726358, 'lambda': 2.089343452610327, 'alpha': 0.2216582281615626}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 08:52:57,663] Trial 90 finished with value: 0.9866298855257383 and parameters: {'n_estimators': 2044, 'learning_rate': 0.10479012164573305, 'max_depth': 14, 'subsample': 0.7775291559656712, 'colsample_bytree': 0.6875738504840576, 'gamma': 0.26462756506868323, 'lambda': 3.2238438502608573, 'alpha': 0.07981999978952944}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 09:00:04,568] Trial 91 finished with value: 0.9866135750723863 and parameters: {'n_estimators': 2085, 'learning_rate': 0.10171044025175913, 'max_depth': 14, 'subsample': 0.7737899700569111, 'colsample_bytree': 0.6856646025664306, 'gamma': 0.2596775465810603, 'lambda': 2.5211976532928575, 'alpha': 0.6979994531633891}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 09:07:00,396] Trial 92 finished with value: 0.9864392125095455 and parameters: {'n_estimators': 2002, 'learning_rate': 0.10064060255462089, 'max_depth': 14, 'subsample': 0.7758102415046687, 'colsample_bytree': 0.6674209712093528, 'gamma': 0.2648523129379107, 'lambda': 3.1511099131218483, 'alpha': 0.5699559733468282}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 09:12:57,083] Trial 93 finished with value: 0.9859806393042425 and parameters: {'n_estimators': 1962, 'learning_rate': 0.08003386239775406, 'max_depth': 13, 'subsample': 0.7573323201778254, 'colsample_bytree': 0.6876878819818795, 'gamma': 0.15225637411703574, 'lambda': 0.5016558885290245, 'alpha': 0.09353510744153831}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 09:19:32,090] Trial 94 finished with value: 0.985744196069138 and parameters: {'n_estimators': 2034, 'learning_rate': 0.17575482131880465, 'max_depth': 14, 'subsample': 0.7498337421969538, 'colsample_bytree': 0.7141556593742027, 'gamma': 0.23197917796179074, 'lambda': 9.679395197169871, 'alpha': 0.9691703141811848}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 09:27:53,180] Trial 95 finished with value: 0.9861426313431246 and parameters: {'n_estimators': 2098, 'learning_rate': 0.06154797184801452, 'max_depth': 14, 'subsample': 0.7698640626837148, 'colsample_bytree': 0.7327717728709374, 'gamma': 0.21084028710479824, 'lambda': 1.1591352156904058, 'alpha': 2.354031040179216}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 09:33:28,111] Trial 96 finished with value: 0.9852043358169594 and parameters: {'n_estimators': 2170, 'learning_rate': 0.1115831491564787, 'max_depth': 12, 'subsample': 0.7797786968120014, 'colsample_bytree': 0.7542019403997962, 'gamma': 0.10337816788784493, 'lambda': 6.350334520442132, 'alpha': 0.4373557159444159}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 09:40:53,746] Trial 97 finished with value: 0.9858185537659667 and parameters: {'n_estimators': 2077, 'learning_rate': 0.07515126537049598, 'max_depth': 14, 'subsample': 0.7169396890905966, 'colsample_bytree': 0.6196667400608978, 'gamma': 0.32023409514294693, 'lambda': 3.26956217115763, 'alpha': 0.06064777155003413}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 09:45:29,710] Trial 98 finished with value: 0.9859637088074584 and parameters: {'n_estimators': 1853, 'learning_rate': 0.14209817266815367, 'max_depth': 13, 'subsample': 0.7958990412861797, 'colsample_bytree': 0.6303277083448157, 'gamma': 0.28394924595160875, 'lambda': 0.9288435360883431, 'alpha': 0.14144557197854313}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 09:53:33,386] Trial 99 finished with value: 0.9855511701119548 and parameters: {'n_estimators': 2234, 'learning_rate': 0.09268786708402073, 'max_depth': 15, 'subsample': 0.7469914595025045, 'colsample_bytree': 0.7108035106237939, 'gamma': 0.3322798520143788, 'lambda': 0.5050832888043156, 'alpha': 6.242454704332096}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 10:01:50,074] Trial 100 finished with value: 0.986289858753636 and parameters: {'n_estimators': 2125, 'learning_rate': 0.04200907837352452, 'max_depth': 14, 'subsample': 0.8329071312294485, 'colsample_bytree': 0.6536454025236577, 'gamma': 0.24901575030953532, 'lambda': 0.09180035536520285, 'alpha': 0.7675060895748556}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 10:08:06,795] Trial 101 finished with value: 0.9868783493294956 and parameters: {'n_estimators': 2382, 'learning_rate': 0.09055103705019289, 'max_depth': 15, 'subsample': 0.8121790993635442, 'colsample_bytree': 0.698974993317169, 'gamma': 0.357114046146373, 'lambda': 2.4875769395997898, 'alpha': 0.31047181752204}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 10:16:32,296] Trial 102 finished with value: 0.9868741352932636 and parameters: {'n_estimators': 2487, 'learning_rate': 0.05516904307959775, 'max_depth': 15, 'subsample': 0.824296265029483, 'colsample_bytree': 0.7246029515922061, 'gamma': 0.37562722915432856, 'lambda': 2.0316648530275634, 'alpha': 0.2776614476738798}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 10:25:26,365] Trial 103 finished with value: 0.9870612380183564 and parameters: {'n_estimators': 2487, 'learning_rate': 0.05029316195629177, 'max_depth': 15, 'subsample': 0.8184560873262151, 'colsample_bytree': 0.7253303357644955, 'gamma': 0.3709629946782507, 'lambda': 1.1469720279329116, 'alpha': 0.20262206280947828}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 10:33:45,254] Trial 104 finished with value: 0.9870047789250006 and parameters: {'n_estimators': 2494, 'learning_rate': 0.050218387667782215, 'max_depth': 15, 'subsample': 0.8429979194390597, 'colsample_bytree': 0.7468030306797847, 'gamma': 0.3856942814240154, 'lambda': 1.3032521778109414, 'alpha': 0.23496442424779107}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 10:43:29,442] Trial 105 finished with value: 0.9866689890923854 and parameters: {'n_estimators': 2484, 'learning_rate': 0.035729585455759105, 'max_depth': 15, 'subsample': 0.8422132227340449, 'colsample_bytree': 0.7494699094072396, 'gamma': 0.3787698585645067, 'lambda': 1.2017806659712291, 'alpha': 0.2009115559771003}. Best is trial 81 with value: 0.9871777819748337.
[I 2025-05-18 10:52:06,206] Trial 106 finished with value: 0.987247534514811 and parameters: {'n_estimators': 2487, 'learning_rate': 0.04967551548387531, 'max_depth': 15, 'subsample': 0.8252641200539966, 'colsample_bytree': 0.7737740714037915, 'gamma': 0.34952521528481467, 'lambda': 0.7330641062318338, 'alpha': 0.31425756893058554}. Best is trial 106 with value: 0.987247534514811.
[I 2025-05-18 11:00:32,331] Trial 107 finished with value: 0.9871953862038371 and parameters: {'n_estimators': 2500, 'learning_rate': 0.05135727332977585, 'max_depth': 15, 'subsample': 0.8080345676703272, 'colsample_bytree': 0.7609074018362313, 'gamma': 0.3515416753330241, 'lambda': 0.8120460332508996, 'alpha': 1.794766097403972e-08}. Best is trial 106 with value: 0.987247534514811.
[I 2025-05-18 11:10:53,776] Trial 108 finished with value: 0.9858443021030191 and parameters: {'n_estimators': 2489, 'learning_rate': 0.02735849299039239, 'max_depth': 15, 'subsample': 0.8244352904002, 'colsample_bytree': 0.7700649698213049, 'gamma': 0.4039536538433286, 'lambda': 1.5585783260062744, 'alpha': 5.32728899315881e-08}. Best is trial 106 with value: 0.987247534514811.
[I 2025-05-18 11:19:00,579] Trial 109 finished with value: 0.9872050350737241 and parameters: {'n_estimators': 2407, 'learning_rate': 0.04932215623463175, 'max_depth': 15, 'subsample': 0.850383727884743, 'colsample_bytree': 0.7781178560883784, 'gamma': 0.35055445877383673, 'lambda': 0.6408887944589198, 'alpha': 1.0326888638598633e-08}. Best is trial 106 with value: 0.987247534514811.
[I 2025-05-18 11:27:53,520] Trial 110 finished with value: 0.9871878553789286 and parameters: {'n_estimators': 2400, 'learning_rate': 0.04662812740277815, 'max_depth': 15, 'subsample': 0.8481553315115857, 'colsample_bytree': 0.7607239430434414, 'gamma': 0.33901596798477257, 'lambda': 0.7893076133458468, 'alpha': 1.3464579385839153e-08}. Best is trial 106 with value: 0.987247534514811.
[I 2025-05-18 11:37:12,962] Trial 111 finished with value: 0.9873415889410657 and parameters: {'n_estimators': 2397, 'learning_rate': 0.04899742663656813, 'max_depth': 15, 'subsample': 0.8424042214542694, 'colsample_bytree': 0.7799914949111625, 'gamma': 0.2997531858054152, 'lambda': 0.8283606956491497, 'alpha': 1.1699720328135459e-08}. Best is trial 111 with value: 0.9873415889410657.
[I 2025-05-18 11:48:59,038] Trial 112 finished with value: 0.9868264208405779 and parameters: {'n_estimators': 2413, 'learning_rate': 0.03356974649920845, 'max_depth': 15, 'subsample': 0.849949695604873, 'colsample_bytree': 0.7772133685186248, 'gamma': 0.34185944842244376, 'lambda': 0.6640403292618899, 'alpha': 1.0090354816077498e-08}. Best is trial 111 with value: 0.9873415889410657.
[I 2025-05-18 11:58:42,328] Trial 113 finished with value: 0.9871991341602457 and parameters: {'n_estimators': 2437, 'learning_rate': 0.043183629590133284, 'max_depth': 15, 'subsample': 0.8788554150206154, 'colsample_bytree': 0.75958519636032, 'gamma': 0.3220913457586469, 'lambda': 0.8413233467785995, 'alpha': 2.0828629244109084e-08}. Best is trial 111 with value: 0.9873415889410657.
[I 2025-05-18 12:07:05,939] Trial 114 finished with value: 0.9872743516761133 and parameters: {'n_estimators': 2451, 'learning_rate': 0.050059232587786905, 'max_depth': 15, 'subsample': 0.8770100404036086, 'colsample_bytree': 0.7580336938894974, 'gamma': 0.31738144388435063, 'lambda': 0.9348650951762247, 'alpha': 1.9265353325280373e-08}. Best is trial 111 with value: 0.9873415889410657.
[I 2025-05-18 12:15:47,643] Trial 115 finished with value: 0.9873733959733606 and parameters: {'n_estimators': 2448, 'learning_rate': 0.048462394509859195, 'max_depth': 15, 'subsample': 0.8765098230090178, 'colsample_bytree': 0.7589625687982171, 'gamma': 0.2988190688855731, 'lambda': 0.9010994701453632, 'alpha': 2.3652859971952107e-08}. Best is trial 115 with value: 0.9873733959733606.
[I 2025-05-18 12:24:57,278] Trial 116 finished with value: 0.9873346757320903 and parameters: {'n_estimators': 2441, 'learning_rate': 0.04325291599150339, 'max_depth': 15, 'subsample': 0.8736062650689151, 'colsample_bytree': 0.7816737097361934, 'gamma': 0.2955845393875145, 'lambda': 0.6516765617783813, 'alpha': 1.858896060740952e-08}. Best is trial 115 with value: 0.9873733959733606.
[I 2025-05-18 12:33:59,384] Trial 117 finished with value: 0.9873214788272776 and parameters: {'n_estimators': 2446, 'learning_rate': 0.043988544355184685, 'max_depth': 15, 'subsample': 0.874450176207771, 'colsample_bytree': 0.7812882237256196, 'gamma': 0.30058427640145713, 'lambda': 0.7610416035228766, 'alpha': 2.1063198587268283e-08}. Best is trial 115 with value: 0.9873733959733606.
[I 2025-05-18 12:45:00,921] Trial 118 finished with value: 0.985429426511102 and parameters: {'n_estimators': 2442, 'learning_rate': 0.02156364889453994, 'max_depth': 15, 'subsample': 0.8565247638961231, 'colsample_bytree': 0.7571342497046983, 'gamma': 0.3003312737174191, 'lambda': 0.6598140598422675, 'alpha': 2.3888825242441383e-08}. Best is trial 115 with value: 0.9873733959733606.
[I 2025-05-18 12:54:12,811] Trial 119 finished with value: 0.9873873053765879 and parameters: {'n_estimators': 2404, 'learning_rate': 0.04164054688849172, 'max_depth': 15, 'subsample': 0.8742551277110467, 'colsample_bytree': 0.7820257997337541, 'gamma': 0.28577573432282644, 'lambda': 0.2644763213229494, 'alpha': 2.7619820968811404e-08}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 13:04:52,284] Trial 120 finished with value: 0.9861844829957703 and parameters: {'n_estimators': 2454, 'learning_rate': 0.02660343124111994, 'max_depth': 15, 'subsample': 0.8745176506511457, 'colsample_bytree': 0.78033845176588, 'gamma': 0.2951653145833996, 'lambda': 0.8152263255514167, 'alpha': 2.0930564594980576e-08}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 13:13:17,394] Trial 121 finished with value: 0.9871726342823443 and parameters: {'n_estimators': 2400, 'learning_rate': 0.042377412681157106, 'max_depth': 15, 'subsample': 0.8889698437255684, 'colsample_bytree': 0.7597865888184572, 'gamma': 0.3338505716605857, 'lambda': 0.2508176397692741, 'alpha': 4.2377954374123146e-08}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 13:21:42,377] Trial 122 finished with value: 0.98725536544899 and parameters: {'n_estimators': 2399, 'learning_rate': 0.043094084364884475, 'max_depth': 15, 'subsample': 0.8860625780260715, 'colsample_bytree': 0.765942856264843, 'gamma': 0.32983991277763913, 'lambda': 0.27658361029269923, 'alpha': 3.7467761736554514e-08}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 13:31:01,214] Trial 123 finished with value: 0.987196820725242 and parameters: {'n_estimators': 2318, 'learning_rate': 0.039479087873652216, 'max_depth': 15, 'subsample': 0.8797727723556815, 'colsample_bytree': 0.7920245966512272, 'gamma': 0.2851553876640029, 'lambda': 0.5689572948607535, 'alpha': 1.8007343253339606e-07}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 13:40:46,120] Trial 124 finished with value: 0.9869741129837736 and parameters: {'n_estimators': 2314, 'learning_rate': 0.03565857030117601, 'max_depth': 15, 'subsample': 0.8752470557234667, 'colsample_bytree': 0.7886881559655927, 'gamma': 0.23844103394396693, 'lambda': 0.5276146074590543, 'alpha': 1.7229335377280579e-07}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 13:52:21,480] Trial 125 finished with value: 0.9852533740872532 and parameters: {'n_estimators': 2414, 'learning_rate': 0.01885347704420024, 'max_depth': 15, 'subsample': 0.8864878712790922, 'colsample_bytree': 0.8126554967534008, 'gamma': 0.22066058935118793, 'lambda': 0.14743579779074314, 'alpha': 1.5336025535430775e-08}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 14:02:35,418] Trial 126 finished with value: 0.9866357420763399 and parameters: {'n_estimators': 2437, 'learning_rate': 0.030546179168307697, 'max_depth': 15, 'subsample': 0.9135029164254239, 'colsample_bytree': 0.7697766941823476, 'gamma': 0.2818028055596106, 'lambda': 0.9257378942307876, 'alpha': 3.5809665715540006e-08}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 14:11:26,991] Trial 127 finished with value: 0.9863188494499348 and parameters: {'n_estimators': 2323, 'learning_rate': 0.04445279912065863, 'max_depth': 15, 'subsample': 0.9027934537347162, 'colsample_bytree': 0.7958504632808088, 'gamma': 0.309689356833891, 'lambda': 5.294548811792661, 'alpha': 1.0704266210760984e-07}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 14:20:24,704] Trial 128 finished with value: 0.9872263527973527 and parameters: {'n_estimators': 2455, 'learning_rate': 0.0401657511498772, 'max_depth': 15, 'subsample': 0.8509223156803895, 'colsample_bytree': 0.7843777964527353, 'gamma': 0.34188504740671266, 'lambda': 7.299764791985171e-07, 'alpha': 6.406158356333908e-08}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 14:29:17,490] Trial 129 finished with value: 0.9873385669204049 and parameters: {'n_estimators': 2464, 'learning_rate': 0.04004312901592535, 'max_depth': 15, 'subsample': 0.9274226188665118, 'colsample_bytree': 0.7809816478306798, 'gamma': 0.27797006305499655, 'lambda': 1.3824496012094897e-05, 'alpha': 6.898161489242479e-08}. Best is trial 119 with value: 0.9873873053765879.
[I 2025-05-18 14:38:53,265] Trial 130 finished with value: 0.9875640292867937 and parameters: {'n_estimators': 2276, 'learning_rate': 0.03844051055554991, 'max_depth': 15, 'subsample': 0.9240462730295915, 'colsample_bytree': 0.7843854088875306, 'gamma': 0.17870057468056816, 'lambda': 1.401713227770493e-05, 'alpha': 2.535993244900807e-07}. Best is trial 130 with value: 0.9875640292867937.
[I 2025-05-18 14:48:35,788] Trial 131 finished with value: 0.9878166629720316 and parameters: {'n_estimators': 2459, 'learning_rate': 0.03999996560758967, 'max_depth': 15, 'subsample': 0.9376778881209966, 'colsample_bytree': 0.7855003885822004, 'gamma': 0.19449049002132435, 'lambda': 4.458692856679572e-06, 'alpha': 2.922184313953772e-07}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 14:58:59,175] Trial 132 finished with value: 0.9875111561866537 and parameters: {'n_estimators': 2456, 'learning_rate': 0.03397302051426409, 'max_depth': 15, 'subsample': 0.9423348083005482, 'colsample_bytree': 0.8199401568918144, 'gamma': 0.1952428836784086, 'lambda': 8.120407419111717e-07, 'alpha': 6.843337305113291e-08}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 15:09:51,059] Trial 133 finished with value: 0.9868937991559037 and parameters: {'n_estimators': 2462, 'learning_rate': 0.027448120944972912, 'max_depth': 15, 'subsample': 0.9670784248908393, 'colsample_bytree': 0.8198078670515636, 'gamma': 0.190893305581402, 'lambda': 1.2438977017701964e-06, 'alpha': 4.820237965511705e-07}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 15:20:18,843] Trial 134 finished with value: 0.9873553402200681 and parameters: {'n_estimators': 2409, 'learning_rate': 0.03207817730781709, 'max_depth': 15, 'subsample': 0.9224313355139376, 'colsample_bytree': 0.7827022504159105, 'gamma': 0.15859669542320942, 'lambda': 6.238342490258195e-06, 'alpha': 7.193770682403063e-08}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 15:30:25,654] Trial 135 finished with value: 0.9873432603905432 and parameters: {'n_estimators': 2350, 'learning_rate': 0.033597663880353054, 'max_depth': 15, 'subsample': 0.9387053613651503, 'colsample_bytree': 0.786700172840062, 'gamma': 0.16370615842369135, 'lambda': 5.930080868194264e-06, 'alpha': 7.279312291687852e-08}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 15:40:59,107] Trial 136 finished with value: 0.9871683544633468 and parameters: {'n_estimators': 2333, 'learning_rate': 0.0319070474093166, 'max_depth': 15, 'subsample': 0.9346240078249235, 'colsample_bytree': 0.801308852046712, 'gamma': 0.16568741005452628, 'lambda': 8.990258104580535e-06, 'alpha': 8.616595317729572e-08}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 15:52:32,246] Trial 137 finished with value: 0.9860217330757822 and parameters: {'n_estimators': 2287, 'learning_rate': 0.023252582106261088, 'max_depth': 15, 'subsample': 0.9470215001458838, 'colsample_bytree': 0.8123115479832995, 'gamma': 0.11730561606480666, 'lambda': 3.683507707819755e-06, 'alpha': 3.094561789021736e-08}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 16:02:53,993] Trial 138 finished with value: 0.9874592647043257 and parameters: {'n_estimators': 2361, 'learning_rate': 0.035034276682463554, 'max_depth': 15, 'subsample': 0.9687638741546695, 'colsample_bytree': 0.7735869715391734, 'gamma': 0.1344862817615808, 'lambda': 7.484834801922105e-06, 'alpha': 3.327642083293211e-07}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 16:11:33,322] Trial 139 finished with value: 0.9862154752549328 and parameters: {'n_estimators': 2356, 'learning_rate': 0.034823076986971915, 'max_depth': 14, 'subsample': 0.9809107396163881, 'colsample_bytree': 0.7782857204115096, 'gamma': 0.09364964615004176, 'lambda': 1.7386924904069253e-05, 'alpha': 4.478044362754981e-07}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 16:22:24,155] Trial 140 finished with value: 0.9867319016816388 and parameters: {'n_estimators': 2288, 'learning_rate': 0.02820617049724375, 'max_depth': 15, 'subsample': 0.9220738947037868, 'colsample_bytree': 0.7878299970146798, 'gamma': 0.04970241955997198, 'lambda': 4.0242764284087495e-06, 'alpha': 2.902801022161066e-07}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 16:32:54,008] Trial 141 finished with value: 0.9877612947396224 and parameters: {'n_estimators': 2390, 'learning_rate': 0.03755940684099901, 'max_depth': 15, 'subsample': 0.943702196280772, 'colsample_bytree': 0.8315195128733719, 'gamma': 0.12442280137428627, 'lambda': 8.258780162522e-06, 'alpha': 7.092090163423286e-08}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 16:43:25,935] Trial 142 finished with value: 0.987756125892131 and parameters: {'n_estimators': 2389, 'learning_rate': 0.0373441056465176, 'max_depth': 15, 'subsample': 0.9426169397849775, 'colsample_bytree': 0.8348890289993507, 'gamma': 0.13003327838511325, 'lambda': 2.6456584450698526e-05, 'alpha': 6.437586239609914e-08}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 16:54:44,233] Trial 143 finished with value: 0.9864147891328989 and parameters: {'n_estimators': 2350, 'learning_rate': 0.024295310879278432, 'max_depth': 15, 'subsample': 0.9423729118947062, 'colsample_bytree': 0.8313856736590636, 'gamma': 0.13482278982907792, 'lambda': 1.2196559610239222e-05, 'alpha': 7.112853117320481e-08}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 17:05:18,918] Trial 144 finished with value: 0.9877997156643907 and parameters: {'n_estimators': 2423, 'learning_rate': 0.0373655505938673, 'max_depth': 15, 'subsample': 0.9741381538960191, 'colsample_bytree': 0.8247127012134795, 'gamma': 0.13160653854705914, 'lambda': 3.365644638739895e-05, 'alpha': 2.049957466102124e-06}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 17:14:04,470] Trial 145 finished with value: 0.9864192109672146 and parameters: {'n_estimators': 2393, 'learning_rate': 0.03590065225353421, 'max_depth': 14, 'subsample': 0.9637983217359463, 'colsample_bytree': 0.845786532851908, 'gamma': 0.12089783852998465, 'lambda': 3.357615068180455e-05, 'alpha': 1.0876425549147485e-06}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 17:16:25,270] Trial 146 finished with value: 0.9520480480773821 and parameters: {'n_estimators': 2372, 'learning_rate': 0.03220458428477149, 'max_depth': 6, 'subsample': 0.9778409316628633, 'colsample_bytree': 0.8161226635529553, 'gamma': 0.15455675202524669, 'lambda': 8.236254883764219e-05, 'alpha': 1.7245142136126883e-06}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 17:20:17,142] Trial 147 finished with value: 0.9694623854411079 and parameters: {'n_estimators': 2277, 'learning_rate': 0.01977209319746182, 'max_depth': 10, 'subsample': 0.9985021036329198, 'colsample_bytree': 0.8288023992883803, 'gamma': 0.19651606837289368, 'lambda': 5.3554289492149485e-06, 'alpha': 1.6253714908559552e-07}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 17:30:46,254] Trial 148 finished with value: 0.9877720095900095 and parameters: {'n_estimators': 2419, 'learning_rate': 0.03812689415014961, 'max_depth': 15, 'subsample': 0.9568994265390411, 'colsample_bytree': 0.8039188561598449, 'gamma': 0.13754273020677527, 'lambda': 2.3811273297010285e-06, 'alpha': 2.6723250419185224e-07}. Best is trial 131 with value: 0.9878166629720316.
[I 2025-05-18 17:39:23,468] Trial 149 finished with value: 0.9866988449395176 and parameters: {'n_estimators': 2342, 'learning_rate': 0.03784112700523976, 'max_depth': 14, 'subsample': 0.9536701957054081, 'colsample_bytree': 0.8038416880006265, 'gamma': 0.05066443447540406, 'lambda': 2.360053962726179e-06, 'alpha': 2.8855325077793323e-07}. Best is trial 131 with value: 0.9878166629720316.
"""

def parse_log_data(log_str):
    """解析日志字符串，提取试验数据."""
    trials_data = []
    # 正则表达式匹配包含 "Trial X finished with value: Y and parameters: Z" 的行
    # 我们需要捕获 value 和 parameters
    trial_pattern = re.compile(
        r"Trial \d+ finished with value: ([\d.]+) and parameters: (\{.*?\})\."
    )
    # 有些行可能还包含 datetime 信息，我们也捕获它
    datetime_pattern = re.compile(r"\[I (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\]")

    current_datetime_start = None # Optuna trial 需要 datetime_start 和 datetime_complete

    for line in log_str.strip().split('\n'):
        dt_match = datetime_pattern.search(line)
        if dt_match:
            # 假设日志行中的时间戳是试验的完成时间
            # 我们没有每个trial的开始时间，可以简单地将完成时间也设为开始时间，或者不设置
            # Optuna 会自动处理 datetime_start 和 datetime_complete
            # 为了更精确，如果日志中有开始和结束时间，应该都解析出来
            # 这里我们简单用日志行的时间作为试验的完成时间
            try:
                current_datetime_complete = datetime.strptime(dt_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
            except ValueError:
                logger.warning(f"Could not parse datetime from line: {line}")
                current_datetime_complete = datetime.now() # Fallback
        else:
            # 如果当前行没有时间戳，我们可以沿用上一行的时间戳或设为None
            # 对于add_trial, datetime_start和datetime_complete是可选的
            pass


        match = trial_pattern.search(line)
        if match:
            value_str = match.group(1)
            params_str = match.group(2)
            try:
                value = float(value_str)
                # 将字符串形式的字典转换为真实的Python字典
                # 需要将 'learning_rate': 1.23e-05 这种科学计数法正确处理
                # json.loads 对于标准的JSON字典可以，但Python的repr字符串可能需要ast.literal_eval
                # 或者更简单，如果格式固定，直接用eval（但要注意安全性，这里因为是我们自己的日志，风险较低）
                params = eval(params_str) # 使用eval，因为日志中的字典是Python对象表示

                trials_data.append({
                    'value': value,
                    'params': params,
                    'datetime_complete': current_datetime_complete if 'current_datetime_complete' in locals() else None
                })
                logger.info(f"Parsed trial: Value={value}, Params={params}")
            except Exception as e:
                logger.error(f"Error parsing line: {line}\nError: {e}")
    return trials_data
def main():
    logger.info(f"Starting to populate Optuna DB: {STORAGE_URL} for study: {STUDY_NAME}")

    # 1. 解析日志数据
    parsed_trials = parse_log_data(log_data)
    if not parsed_trials:
        logger.warning("No trials parsed from log_data. Exiting.")
        return

    # 2. 定义参数的分布 (保持不变)
    param_distributions = {
        'n_estimators': IntDistribution(low=500, high=2500),
        'learning_rate': FloatDistribution(low=0.001, high=0.5, log=True),
        'max_depth': IntDistribution(low=3, high=15),
        'subsample': FloatDistribution(low=0.5, high=1.0),
        'colsample_bytree': FloatDistribution(low=0.5, high=1.0),
        'gamma': FloatDistribution(low=0.0, high=1.0),
        'lambda': FloatDistribution(low=1e-8, high=10.0, log=True),
        'alpha': FloatDistribution(low=1e-8, high=10.0, log=True),
    }
    logger.info(f"Defined parameter distributions: {param_distributions}")

    # 3. 创建或加载 Optuna study (保持不变)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize", # 假设是最大化AUC
        load_if_exists=True
    )
    logger.info(f"Study '{STUDY_NAME}' created/loaded from {STORAGE_URL}.")
    logger.info(f"Initial number of trials in study: {len(study.trials)}")

    # 4. 将解析的试验添加到 study 中 (修正部分)
    added_count = 0
    skipped_count = 0
    # 为了更准确地避免重复，我们可以检查参数和值的组合
    existing_trials_summary = set()
    for t in study.trials:
        # 将参数字典转换为可哈希的元组，以便存入set
        param_tuple = tuple(sorted(t.params.items()))
        existing_trials_summary.add((param_tuple, t.value))

    for trial_data in parsed_trials:
        params = trial_data['params']
        value = trial_data['value']
        # dt_complete = trial_data['datetime_complete'] # 不再直接使用

        # 检查这个参数组合和值是否已存在
        param_tuple_to_check = tuple(sorted(params.items()))
        if (param_tuple_to_check, value) in existing_trials_summary:
            logger.info(f"Skipping duplicate trial (params and value match): {params} -> {value}")
            skipped_count += 1
            continue

        current_trial_distributions = {}
        valid_params = True
        for p_name in params.keys(): # 只需检查日志中存在的参数
            if p_name in param_distributions:
                current_trial_distributions[p_name] = param_distributions[p_name]
            else:
                logger.warning(f"Parameter '{p_name}' from log (value: {params[p_name]}) is not in defined distributions. This parameter will be passed but Optuna might not use its distribution info if it needs to resample it.")
                # Optuna在add_trial时，如果distributions字典中没有对应参数的分布，
                # 它可能会尝试推断或在后续采样中忽略该参数的特定分布知识。
                # 对于已经完成的试验，参数值本身是主要的。
        
        # 如果希望严格要求所有参数都有分布定义，可以在这里加判断
        # if not all(p in current_trial_distributions for p in params.keys()):
        #     logger.warning(f"Skipping trial due to missing distribution for some params: {params}")
        #     continue

        try:
            # ***** 关键修正：移除了 datetime_start 和 datetime_complete *****
            trial_to_add = optuna.trial.create_trial(
                params=params,
                distributions=current_trial_distributions,
                value=value,
                state=optuna.trial.TrialState.COMPLETE
                # user_attrs 和 system_attrs 如果有也可以添加
            )
            study.add_trial(trial_to_add)
            # 添加到检查集合中，防止日志内部本身有完全重复的行被重复添加
            existing_trials_summary.add((param_tuple_to_check, value))
            logger.info(f"Successfully added trial: {params} -> {value}")
            added_count +=1
        except Exception as e:
            logger.error(f"Error adding trial {params} -> {value} to study: {e}")
            logger.error(f"Distributions used: {current_trial_distributions}")

    logger.info(f"Finished populating database. Added {added_count} new trials. Skipped {skipped_count} duplicate/invalid trials.")
    logger.info(f"Total trials in study '{STUDY_NAME}' now: {len(study.trials)}")
    if study.trials:
        try:
            logger.info(f"Best trial so far: {study.best_trial.params} with value {study.best_trial.value}")
        except ValueError: # 如果所有试验都失败了，可能没有best_trial
            logger.warning("Could not determine best trial, possibly all trials failed or are ongoing.")
    else:
        logger.warning("No trials in study after population attempt.")


if __name__ == "__main__":
    main()