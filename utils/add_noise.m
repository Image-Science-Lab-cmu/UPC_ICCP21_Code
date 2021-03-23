function [out] = add_noise(x, time, sensor)
% % add noise to signal x [0 to well-capacity]
% x     : signal <photon counts per second>
% time  : exposure time (s)
% sensor.capacity: full-well-capacity
% sensor.read_std: std of read-out noise
% sensor.gain: 1/L

out = poissrnd(x * time) + sensor.noise_std * randn(size(x));
out = min(out, sensor.capacity);
out = out * sensor.gain;

end