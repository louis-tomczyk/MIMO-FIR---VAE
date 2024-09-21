rst
x = readmatrix("ma u_g - mimo cma - lr 10.00 - Rs 64 - mod 64QAM - nu 0.025 - SNRdB 50 - PhEnd 14 - Sph 0.6 - Th_in 0.0 - Nplts 25 - fpol 1.0 - NSbB 250 - NSbF 20.0 - SNR_dB 25 - NspT 13.csv");
WinSize = x(:,1);
BER_u = x(:,2);
BER_g = x(:,3);

figure
    hold on
    plot(WinSize,BER_u)
    plot(WinSize,BER_g)
    xlabel('window size [samples]')
    ylabel('BER')
    title('Impact of window size on the moving average for CPR')
    xlabel('window size [batches]')
    legend('$\phi_{end} = 14 [deg], f_{pol} = 1 [kHz]$, ma u',...)
    '$\phi_{end} = 14 [deg], f_{pol} = 1 [kHz]$, ma g')
    axis square
    set(gcf, 'Position', [0.0198,0.0009,0.5255,0.8824])
    legend boxoff
    box on