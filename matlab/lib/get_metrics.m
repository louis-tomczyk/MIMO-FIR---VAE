

function metrics = get_metrics(thetas_est,thetas_gnd,caps)


Err                       = thetas_est-thetas_gnd(3,:);   % [deg]
metrics.ErrMean(caps.kdata,1)  = mean(Err);
metrics.ErrStd(caps.kdata,1)   = std(Err);
metrics.ErrRms(caps.kdata,1)   = metrics.ErrStd(caps.kdata)/metrics.ErrMean(caps.kdata);

Rs              = get_value_from_filename(caps.myInitPath,'Rs',caps.Fn{1});
if Rs == 64
    Err         = [zeros(1,20),Err];
elseif Rs == 128
    Err         = [zeros(1,40),Err];
end

metrics.Err = Err;

params_avg.av_method    = "mirror";
params_avg.av_period    = 5;
params_std.method       = "mirror";
params_std.period       = 5;

metrics.Err_mov_avg = moving_average(metrics.Err,params_avg);
metrics.Err_mov_std = moving_std(metrics.Err,params_std);



