x = readmatrix("Entropy - Lambda 16-32-64 QAM.csv");

H       = x(:,1);
xl64    = x(:,2);
xl32    = x(:,3);
xl16    = x(:,4);

xl64    = xl64(~isnan(xl64));
xl32    = xl32(~isnan(xl32));
xl16    = xl16(~isnan(xl16));

Nl64    = length(xl64);
Nl32    = length(xl32);
Nl16    = length(xl16);

H64     = H(1:Nl64);
H32     = H(1:Nl32);
H16     = H(1:Nl16);

figure
semilogy(H64,xl64)
hold on
semilogy(H32,xl32)
semilogy(H16,xl16)