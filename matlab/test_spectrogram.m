Fs=22050;
if 0
    N=10000;
    n=0:N-1;
    x=[cos(63*pi/64*n) cos(2*pi/8*n)];
else
    Ts=1/Fs;
    t=0:Ts:2;                    % 2 secs
    x=chirp(t,0,2,Fs/2);       % Start @ 100Hz, cross 200Hz at t=1sec
end
Nwin=256;
window=hamming(Nwin);
num_overlap=Nwin/2;


spectrogram(x,window,num_overlap,[],Fs,'yaxis')

audiowrite("C:\ak\Work\mayron_bird\train_audio\asbfly\aaa_test.ogg",x,Fs)