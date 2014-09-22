%~ to run: 'source template_regression.m'
%~ to run from command line, passing arrayrices Xarray and Yarray
%~ octave --eval "global Xarray = {[-.25, 0, .25]; [-.25, .25]}; global Yarray = {[1, 0, 1]; [1, 1]}; source template_regression.m"
%~ clear
function Y = template_fn(X, params) % a, b, s
  Xs = params(3) .* X;
  Y = params(1) .* (exp(-(Xs-1) .^2) + exp(-(Xs+1) .^2)) + params(2) .* exp(-Xs .^2);
  %~ article: a * (exp(-xsm*xsm) + exp(-xsp*xsp)) + b * exp(-x*x)
  %~ Y = params(1) .* (exp(-(X - params(3)) .^2) + exp(-(X + params(3)) .^2)) + params(2) .* exp(-X .^2);
endfunction

function d = error_fn(params, X, Y)
  d = sumsq( ( template_fn(X, params) - Y ) );
  %~ params, d
endfunction

% arg: X, Y: row vectors of floats
%~ http://www.gnu.org/software/octave/doc/interpreter/Minimizers.html
%~ http://www.arrayhworks.es/es/help/optim/ug/fminunc-unconstrained-minimization.html
%~ http://stackoverflow.com/questions/17598717/output-function-for-fminunc-in-octave
function [success, learntparams, error] = optimize(X, Y)
  startparams=[.25, .25, 5]; % average of min and max for each param
  [learntparams, error, success, out] = fminunc ( @(params)(error_fn(params, X, Y)), startparams);
  learntparams, error, success, out

  %~ plot (X, Y, 'cr') ; % plot input pts
  %~ hold on ; Xplot = [-.25 : .01 : .25]'; plot (Xplot, template_fn(Xplot, learntparams), 'cb'); % plot learned template
  %~ legend('input', 'learntparams');
endfunction

% arg: Xarray, Yarray: each plane is one row
function [success, bestparams, bestslice, besterror] = optimize_array(Xarray, Yarray)
  besterror = 1E10; bestparams = 0; bestslice = 0; success = 0;
  for row = 1:size(Xarray, 1)
    row
    [localsuccess, learntparams, error] = optimize(Xarray{row}, Yarray{row});
    if (localsuccess && besterror > error)
      success = localsuccess;
      besterror = error;
      bestslice = row-1; % start indexing at 0 for C++
      bestparams = learntparams;
    endif
  endfor
endfunction

hold off ;
%~ real optimization
%~ X = [1, 2, 3]; Y = [3, 4, 5];
%~ expectedparams=[.2, .1, 5];
%~ expectedparams=[rand(), rand(), 10*rand()]
%~ X = [-.25 : .01 : .25]; Y = template_fn(X, expectedparams);
%~ X, Y
%~ [success, learntparams, error] = optimize(X, Y);
%~ error=norm(Y-template_fn(X, learntparams))

%~ Xarray = {[-.25, 0, .25]; [-.25, .25]}; Yarray = {[1, 0, 1]; [1, 1]}; % a sample test
global Xarray; global Yarray; % thus Xarray and Yarray can be passed by command line
Xarray, Yarray
[success, bestparams, bestslice, besterror] = optimize_array(Xarray, Yarray);
success, bestparams, bestslice, besterror
