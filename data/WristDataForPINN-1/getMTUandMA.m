function [MTU,MA] = getMTUandMA(AngleInRad)
FCR_mtu_reg  = @(x) 0.2982  + 0.01423   *cos(x*0.3874)  - 0.03854 *sin(x*0.3874);
FCU_mtu_reg  = @(x) 0.2628  + 0.04982   *cos(x*0.2454 ) - 0.06095 *sin(x*0.2454);
ECRL_mtu_reg = @(x) -0.1317 + 0.4661    *cos(x*0.07499) + 0.1343  *sin(x*0.07499);
ECRB_mtu_reg = @(x) 0.2371  + 0.04893   *cos(x*0.2138)  + 0.0622  *sin(x*0.2138);
ECU_mtu_reg  = @(x) 0.2905  + -0.0001401*cos(x*1.213)   + 0.005483*sin(x*1.213);

FCR_ma_reg  = @(x) - (3732599*cos((1937*x)/5000))/250000000 - (2756351*sin((1937*x)/5000))/500000000;
FCU_ma_reg  = @(x) - (1495713*cos((1227*x)/5000))/100000000 - (3056457*sin((1227*x)/5000))/250000000;
ECRL_ma_reg = @(x) (10071157*cos((7499*x)/100000))/1000000000 - (34952839*sin((7499*x)/100000))/1000000000;
ECRB_ma_reg = @(x) (332459*cos((1069*x)/5000))/25000000 - (5230617*sin((1069*x)/5000))/500000000;
ECU_ma_reg  = @(x) (3833970711819040923*cos((1213*x)/1000))/576460752303423488000 + (783715917163374201*sin((1213*x)/1000))/4611686018427387904000;


MTU = [FCR_mtu_reg(AngleInRad) FCU_mtu_reg(AngleInRad) ECRL_mtu_reg(AngleInRad) ECRB_mtu_reg(AngleInRad) ECU_mtu_reg(AngleInRad)];
MA =  [-FCR_ma_reg(AngleInRad) -FCU_ma_reg(AngleInRad) -ECRL_ma_reg(AngleInRad) -ECRB_ma_reg(AngleInRad) -ECU_ma_reg(AngleInRad)];
end