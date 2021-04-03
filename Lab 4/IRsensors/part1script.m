lambda1 = 3.6e-6; % lower limit of the wavelength
lambda2 = 5e-6; % upper limit of the wavelength

n = 1000; % number of terms in the approximation of the integral

c = 2.998e8;  % speed of light [m/s]
h = 6.626e-34; % Planck's constant [Js]
k = 1.381e-23; % Boltzmann's constant [J/K]

Trange = 1:1000;

Intensity = zeros(size(Trange));
delta_lambda = (lambda2 - lambda1) / n;

for p=1:length(Trange),
  T = Trange(p);
  I = 0;
  for lambda=lambda1:delta_lambda:lambda2,
    I = I + 2*pi*h*c^2/(lambda^5*(exp(h*c/(lambda*k*T))-1)) * delta_lambda;
  end
  Intensity(p) = I;
end

figure(1);clf
plot(Trange,Intensity);
