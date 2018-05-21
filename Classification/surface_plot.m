a = ["Alef","Ayin","Bet","Dalet","Gimel","He","Het","Kaf","Kaf-final","Lamed","Mem","Mem-medial","Nun-final","Nun-medial","Pe","Pe-final","Qof","Resh","Samekh","Shin","Taw","Tet","Tsadi-final","Tsadi-medial","Waw","Yod","Zayin"]

x = 1:29;
y = 1:27;
[X,Y] = meshgrid(x,y);

[X_q,Y_q] = meshgrid(1:0.1:29,1:0.1:27);
P_q = interp2(X,Y,mat,X_q,Y_q);
surf(X_q,Y_q,P_q,'EdgeColor','None');
%%

heatmap(mat')
ylabel("Sliding window")
xlabel("Letter")
