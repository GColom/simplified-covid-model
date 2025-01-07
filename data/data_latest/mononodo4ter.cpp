#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <assert.h>
#include <cmath> 
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <algorithm>
#include <math.h>
#include <sstream>
#include <time.h>
#include <iomanip>      // std::setprecision
#include <chrono>
#include <random>
//#define PAU system("Pause")
using namespace std;

#define risultati "../risultati"

double min(double a, double b) { return (a<=b) ? a : b; }
double max(double a, double b) { return (a>=b) ? a : b; }

//void Mononodo() {
int main() {

srand((unsigned) time(0));

int ii, jj;

//parametri

const double dt=1.0/24.0;  // passo temporale [d^-1]

const int popolazione_totale=(1020096-133205); // popolazione totale

const int inizio_epidemia=44; // inizio dell'epidemia simulata (giorno 0): 13 febbraio 2020

const int tmax=(int)round((double)(365*3.5-inizio_epidemia)/dt);  // tempo finale [d]

double betaI=dt/1.2;  // probabilità di contagio per gli I
double betaA=betaI;   // probabilità di contagio per gli A

double alfa=0.14;  // frazione di E che diventa I
double delT=0.30;  // frazione di RT che diventa D
double chi=0.30;   // frazione deli anziani (>=60 a) nella popolazione
double chiT=0.70;  // frazione degli anziani in RT
double chiH=0.70;  // frazione degli anziani in RH

double Te=2.0/dt;  // tempo medio di vita degli E [d]
double Ti=5.5/dt;  // tempo medio di vita degli I [d]
double Ta=Ti;      // tempo medio di vita degli A [d]
double Tt=10/dt;   // tempo medio di vita degli RT [d]
double Th=10/dt;   // tempo medio di vita degli RH [d]
double Tq=21/dt;   // tempo medio di vita degli RQ [d]
double Tc=Tq;      // tempo medio di vita degli RC [d]
double Tg=180/dt;  // durata media dell'immunità conferita dalla guarigione [d]
double Tv=180/dt;  // durata media dell'immunità conferita dalla vaccinazione [d]

double mobv[]={1.00, 1.00, 1.00, 0.24, 0.16, 0.17, 0.18, 0.24, 0.24, 0.24, 0.34, 0.24, 0.21, 0.27, 0.25, 0.23, 0.29, 0.26, 0.20, 0.39, 0.29, 0.65, 0.49, 0.35, 0.29, 0.24, 0.19, 0.25, 0.06, 0.08, 0.18, 0.21, 0.18, 0.12, 0.09, 0.06, 0.03, 0.02, 0.03, 0.018, 0.027, 0.017};   // mobilità interna (tasso di contatto) + VEDI RIGA 287
double  lex[]={   0,   54,   61,   71,   83,  125,  139,  155,  167,  181,  258,  299,  320,  341,  359,  375,  398,  418,  429,  468,  482,  537,  560,  575,  610,  635,  688,  714,  727,  750,  770,  791,  810,  859,  900,  910,  930,  965,  985,  1007,  1036,  1050, 9999};
// dpcm & c:        23feb 01mar 11mar 23mar 04mag 18mag 03giu 15giu 29giu 14set 25ott 15nov 06dic 24dic 09gen 01feb 21feb 04mar 12apr 26apr

double     Ttv[]={        8,     8,     6,     5,     4,     5,     4,         7,         7,         8,         8,         7,         7,         7,         7,         7,         7,         7,         6,        6,         6,          6,         6,         6,         6,         6,         6,         6,         6,         6,         6,         6,      6,     6,     6,     6,     6,      6,      6,     6,     6,     6,     6,     6,     6,     6,     6,     6,     6,    12,    12,   12,    12, 12};
double     Thv[]={        7,     8,     5,     4,     3,     6,     7,         7,         8,         8,         8,         7,         7,         7,         7,         7,         7,         7,         6,        6,         6,          6,         6,         6,         6,         6,         6,         6,         6,         6,         6,         6,      6,     6,     6,     6,     6,      6,      6,     6,     6,     6,     6,     6,     6,     6,     6,     6,     6,    18,    18,   18,    18, 18};
double   gamTv[]={0.295*0.3, 0.167, 0.301, 0.280, 0.191, 0.185, 0.122, 0.052*0.3, 0.040*0.6, 0.039*0.8, 0.038*0.8, 0.034*1.1, 0.034*0.8, 0.034*0.6, 0.050*0.5, 0.050*0.7, 0.050*1.0, 0.050*1.3, 0.060*1.4, 0.060*2.1, 0.060*1.3, 0.051*2.2, 0.131*1.7, 0.015*1.0, 0.015*1.4, 0.015*1.5, 0.015*1.4, 0.011*1.0, 0.008*1.0, 0.008*1.0, 0.008*0.4, 0.002*1.0,  0.003, 0.003, 0.005, 0.002, 0.002,  0.003,  0.003, 0.002, 0.002, 0.002, 0.002, 0.004, 0.006, 0.002, 0.002, 0.002, 0.005, 0.001, 0.002, 0.003, 0.003, 0.005}; // gammaT
double   gamHv[]={    0.577, 0.885, 1.837, 2.009, 0.584, 0.342, 0.204, 0.157*0.6, 0.130*0.8, 0.134*1.1, 0.140*0.9, 0.122*1.3, 0.122*1.0, 0.122*0.9, 0.156*0.7, 0.156*1.1, 0.156*1.2, 0.156*1.6, 0.184*2.0, 0.184*2.8, 0.184*1.8, 0.191*1.1, 0.265*0.5, 0.071*0.6, 0.108*0.6, 0.145*0.5, 0.118*0.8, 0.099*0.9, 0.052*1.3, 0.052*1.0, 0.052*0.4, 0.012*1.4,  0.027, 0.027, 0.060, 0.032, 0.032,  0.050,  0.050, 0.030, 0.022, 0.022, 0.035, 0.070, 0.200, 0.070, 0.038, 0.030, 0.060, 0.020, 0.020, 0.050, 0.050, 0.060}; // gammaH
double  sintom[]={    0.945, 0.818, 0.647, 0.385, 0.465, 0.589, 0.577,     0.582,     0.606,     0.649,     0.687,     0.725,     0.725,     0.725,     0.719,     0.719,     0.719,     0.719,     0.735,     0.735,     0.735,     0.747,     0.754,     0.894,     0.840,     0.791,     0.790,     0.757,     0.661,     0.661,     0.661,     0.436,  0.436, 0.316, 0.316, 0.239, 0.191,  0.191,  0.175, 0.151, 0.151, 0.141, 0.141, 0.141, 0.119, 0.118, 0.118, 0.083, 0.083, 0.071, 0.071, 0.058, 0.058, 0.058}; // sintomatici
double   morti[]={    0.069, 0.153, 0.304, 0.344, 0.024, 0.019, 0.003,     0.007,     0.026,     0.045,     0.049,     0.025,     0.025,     0.025,     0.027,     0.027,     0.027,     0.027,     0.037,     0.037,     0.037,     0.033,     0.047,     0.013,     0.007,     0.010,     0.018,     0.011,     0.008,     0.005,     0.005,     0.002,  0.004, 0.004, 0.004, 0.004, 0.004,  0.004,  0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004}; // morti
double kalenda[]={        0,    92,   122,   153,   183,   214,   245,       275,       306,       336,       367,       398,       408,       418,       426,       437,       443,       451,       457,       462,       475,       487,       518,       548,       579,       610,       640,       671,       701,       712,       725,       732,    754,   763,   770,   791,   822,     835,   852,   883,   900,   913,   925,   935,   944,   975,   994,  1005,  1030,  1036,  1060,  1066,  1075,  1080, 9999};
//                           01apr  01mag  01giu  01lug  01ago  01set      01ott      01nov      01dic      01gen      01feb                            01mar                                       01apr                            01mag      01giu      01lug      01ago      01set      01ott      01nov      01dic                            01gen          01feb         01mar  01apr           01mag  01giu         01lug                01ago  01set         01ott         01nov         01dic
for(ii=0; ii<sizeof(morti)/sizeof(morti[0]); ii++) morti[ii] *= 0.7;

// varianti
double     kv[]={1.0,      1.56,     1.56*1.60, 5.0*1.56*1.60};
double arrivo[]={  0,       294,           457,           701, 9999};
//                    20ott2020      01apr2021
//                inglese(alfa) indiana(delta)        omicron
int numvar=sizeof(kv)/sizeof(kv[0]);

// piani vaccinali
double pianvac0[]={1997,  85, 1679, 4163, 2493,  569,  520,   82}; // prime e seconde dosi
//double pianvac1[]={   0,   0,    0,    0,    0,    0,   0}; // terza dose
double pianvac1[]={   0,   0,    0,    0,    0, 1595, 5256, 1107}; // terza dose
double kalevacc[]={ 367, 383,  404,  453,  551,  629,  701,  781, 9999}; 
// anno 2021:    01gen
double efficacia1 = 0.5;
for(ii=0; ii<sizeof(pianvac1)/sizeof(pianvac1[0]); ii++) pianvac1[ii] *= efficacia1;
double numeromedici=30000; // personale sanitario con priorità di vaccinazione
double novax=0.10; // frazione di antivaccinisti
bool medicus0, medicus1;
int senex0, senex1;
double giavaccinati0[]={0,0};

double gamC, gamT, gamH, delH, chiQ, chiC=chi, gammaT[2], gammaH[2];

// variabili (I parte)
// 1° indice : tempo
// 2° indice : età (0=giovani; 1=vecchi)
// 3° indice : variante (0=cinese; 1=inglese(alfa); 2=indiana(delta))
vector<double> tt(tmax+1, 0);
vector<vector<double>> N(tmax+1, vector<double>(2, 0)), P(tmax+1, vector<double>(2, 0)), S(tmax+1, vector<double>(2, 0));
vector<vector<vector<double>>> E(tmax+1, vector<vector<double>>(2, vector<double>(numvar, 0))),
                               I(tmax+1, vector<vector<double>>(2, vector<double>(numvar, 0))),
                               A(tmax+1, vector<vector<double>>(2, vector<double>(numvar, 0))),
                               C(tmax+1, vector<vector<double>>(2, vector<double>(numvar, 0)));
vector<vector<double>> RT(tmax+1, vector<double>(2, 0)), RH(tmax+1, vector<double>(2, 0)), RQ(tmax+1, vector<double>(2, 0)),
                       RC(tmax+1, vector<double>(2, 0)), GR(tmax+1, vector<double>(2, 0)), GO(tmax+1, vector<double>(2, 0)),
                       D(tmax+1, vector<double>(2, 0)), V(tmax+1, vector<double>(2, 0));
vector<vector<vector<double>>> SE(tmax+1, vector<vector<double>>(2, vector<double>(numvar, 0))),
                               EI(tmax+1, vector<vector<double>>(2, vector<double>(numvar, 0))),
                               EA(tmax+1, vector<vector<double>>(2, vector<double>(numvar, 0)));
vector<vector<double>> vac0(sizeof(pianvac0)/sizeof(pianvac0[0]), vector<double>(2, 0)),
                       vac1(sizeof(pianvac1)/sizeof(pianvac1[0]), vector<double>(2, 0));

double IRT[numvar], IRH[numvar], IRQ[numvar], ARC[numvar], AGO[numvar], RTD, RTH, RHD, RHQ, RQG, RCG, GRS, GOS, VS, flussoSE[2][numvar];

vector<double> pdfe(tmax,0), pdfi(tmax,0), pdfa(tmax,0), pdfq(tmax,0), pdfc(tmax,0), pdft(tmax,0), pdfh(tmax,0), pdfg(tmax,0), pdfv(tmax,0);
int dismin, Le, Li, La, Lq, Lc, Lt, Lh, Lg, Lv;
double sqm, teta, capp, massimo, somma;

double mob, gente, incI, incA, vaccinabili0, vaccinabili1, vaccinati0[]={0,0}, vaccinati1[]={0,0}, residuo, erret;

int t0, t1, t2, qualekalenda, qualelex, qualevariante, qualevac, nn, vv;

char nomefile[100];

FILE *file1, *file2, *file3, *file4, *file6, *file8, *file9, *nuovofile;

// ricalcolo parametri
for(ii=0; ii<sizeof(lex)/sizeof(lex[0]); ii++) { lex[ii] -= inizio_epidemia; }
for(ii=0; ii<sizeof(kalenda)/sizeof(kalenda[0]); ii++) { kalenda[ii] -= inizio_epidemia; }
for(ii=0; ii<sizeof(kalevacc)/sizeof(kalevacc[0]); ii++) { kalevacc[ii] -= inizio_epidemia; }
for(ii=0; ii<sizeof(arrivo)/sizeof(arrivo[0]); ii++) { arrivo[ii] -= inizio_epidemia; }    
for(ii=0; ii<sizeof(pianvac0)/sizeof(pianvac0[0]); ii++)
    { vac0[ii][0] = pianvac0[ii]*dt*(1.0-chi);  vac0[ii][1] = pianvac0[ii]*dt*chi; }     
for(ii=0; ii<sizeof(pianvac1)/sizeof(pianvac1[0]); ii++)
    { vac1[ii][0] = pianvac1[ii]*dt*(1.0-chi);  vac1[ii][1] = pianvac1[ii]*dt*chi; }    

// distribuzioni statistiche
sqm = 0.1*Te;  teta = sqm*sqm/Te;  capp = Te/teta;
dismin = max(1, round(Te-3*sqm));  Le = (int)round(Te+5*sqm);  somma = 0;
for(ii=1; ii<=Le; ii++) { pdfe[ii] = 0; }
for(ii=dismin; ii<=Le; ii++) { pdfe[ii] = pow(ii,capp-1)*exp(-ii/teta);  somma += pdfe[ii]; }
for(ii=1; ii<=Le; ii++) { pdfe[ii] /= somma; }

sqm = 2.3/dt;  teta = sqm*sqm/Ti;  capp = Ti/teta;
dismin = max(1, round(Ti-3*sqm));  Li = (int)round(Ti+5*sqm);  somma = 0;
for(ii=1; ii<=Li; ii++) { pdfi[ii] = 0; }
for(ii=dismin; ii<=Li; ii++) { pdfi[ii] = pow(ii,capp-1)*exp(-ii/teta);  somma += pdfi[ii]; }
for(ii=1; ii<=Li; ii++) { pdfi[ii] /= somma; }

sqm = 2.3/dt;  teta = sqm*sqm/Ta;  capp = Ta/teta;
dismin = max(1, round(Ta-3*sqm));  La = (int)round(Ta+5*sqm);  somma = 0;
for(ii=1; ii<=La; ii++) { pdfa[ii] = 0; }
for(ii=dismin; ii<=La; ii++) { pdfa[ii] = pow(ii,capp-1)*exp(-ii/teta);  somma += pdfa[ii]; }
for(ii=1; ii<=La; ii++) { pdfa[ii] /= somma; }

sqm = 0.1*Tq;  teta = sqm*sqm/Tq;  capp = Tq/teta;
dismin = max(1, round(Tq-3*sqm));  Lq = (int)round(Tq+5*sqm);  somma = 0;
for(ii=1; ii<=Lq; ii++) { pdfq[ii] = 0; }
for(ii=dismin; ii<=Lq; ii++) { pdfq[ii] = pow(ii,capp-1)*exp(-ii/teta);  somma += pdfq[ii]; }
for(ii=1; ii<=Lq; ii++) { pdfq[ii] /= somma; }

sqm = 0.1*Tc;  teta = sqm*sqm/Tc;  capp = Tc/teta;
dismin = max(1, round(Tc-3*sqm));  Lc = (int)round(Tc+5*sqm);  somma = 0;
for(ii=1; ii<=Lc; ii++) { pdfc[ii] = 0; }
for(ii=dismin; ii<=Lc; ii++) { pdfc[ii] = pow(ii,capp-1)*exp(-ii/teta);  somma += pdfc[ii]; }
for(ii=1; ii<=Lc; ii++) { pdfc[ii] /= somma; }

sqm = 0.1*Tg;  teta = sqm*sqm/Tg;  capp = Tg/teta;
dismin = max(1, round(Tg-3*sqm));  Lg = (int)round(Tg+5*sqm);  massimo = 0;  somma = 0;
for(ii=1; ii<=Lg; ii++) { pdfg[ii] = 0; }
for(ii=dismin; ii<=Lg; ii++) { pdfg[ii] = (capp-1)*log(ii)-ii/teta;  massimo = max(massimo,pdfg[ii]); }
for(ii=dismin; ii<=Lg; ii++) { pdfg[ii] = exp(pdfg[ii]-massimo);  somma += pdfg[ii]; }
for(ii=1; ii<=Lg; ii++) { pdfg[ii] /= somma; }

sqm = 0.1*Tv;  teta = sqm*sqm/Tv;  capp = Tv/teta;
dismin = max(1, round(Tv-3*sqm));  Lv = (int)round(Tv+5*sqm);  massimo = 0;  somma = 0;
for(ii=1; ii<=Lv; ii++) { pdfv[ii] = 0; }
for(ii=dismin; ii<=Lv; ii++) { pdfv[ii] = (capp-1)*log(ii)-ii/teta;  massimo = max(massimo,pdfv[ii]); }
for(ii=dismin; ii<=Lv; ii++) { pdfv[ii] = exp(pdfv[ii]-massimo);  somma += pdfv[ii]; }
for(ii=1; ii<=Lv; ii++) { pdfv[ii] /= somma; }

// variabili (II parte)
// 1° indice : tempo
// 2° indice : età (0=giovani; 1=vecchi)
// 3° indice : variante (0=cinese; 1=inglese(alfa); 2=indiana(delta))
t1 = t2 = 0;
for(ii=0; ii<sizeof(kalenda)/sizeof(kalenda[0]); ii++)
    {
    Tt = round(Ttv[ii]/dt);  sqm = 0.1*Tt;  Lt = round(Tt+5*sqm);  t1 = max(t1,Lt);
    Th = round(Thv[ii]/dt);  sqm = 0.1*Th;  Lh = round(Th+5*sqm);  t2 = max(t2,Lh);
    }
vector<vector<vector<double>>> ge(tmax+1+Le, vector<vector<double>>(2, vector<double>(numvar, 0))),
                               gi(tmax+1+Li, vector<vector<double>>(2, vector<double>(numvar, 0))),
                               ga(tmax+1+La, vector<vector<double>>(2, vector<double>(numvar, 0)));
vector<vector<double>> gt(tmax+1+t1, vector<double>(2, 0)), gh(tmax+1+t2, vector<double>(2, 0)),
                       gq(tmax+1+Lq, vector<double>(2, 0)), gc(tmax+1+Lc, vector<double>(2, 0)),
                       gr(tmax+1+Lg, vector<double>(2, 0)), gn(tmax+1+Lg, vector<double>(2, 0)),
                       gv(tmax+1+Lv, vector<double>(2, 0));

// condizione iniziale
t1 = 0;  tt[t1] = 0;
N[0][0] = round((1.0-chi)*(double)popolazione_totale);  N[0][1] = (double)popolazione_totale-N[0][0];
P[0][0] = N[0][0];  S[0][0] = N[0][0];                  P[0][1] = N[0][1];  S[0][1] = N[0][1];
SE[0][0][0] = 1.0;

// file dei risultati
sprintf(nomefile, "%s/virus_mononodo_info.txt", risultati);  file1 = fopen(nomefile, "w");
for(ii=1; ii<=Lg; ii++) { fprintf(file1, "%d\t%g\n", ii, pdfg[ii]); }
fprintf(file1, "\n\nV\n");
for(ii=1; ii<=Lv; ii++) { fprintf(file1, "%d\t%g\n", ii, pdfv[ii]); }
fclose(file1);

nuovofile = fopen("prova.csv","w");
sprintf(nomefile, "%s/virus_mononodo_popolazioni.csv", risultati);   file2 = fopen(nomefile, "w");
sprintf(nomefile, "%s/virus_mononodo_buffer.csv", risultati);   file3 = fopen(nomefile, "w");
sprintf(nomefile, "%s/virus_mononodo_controllo.csv", risultati);  file4 = fopen(nomefile, "w");
sprintf(nomefile, "%s/virus_mononodo_parametri.txt", risultati);  file6 = fopen(nomefile, "w");
sprintf(nomefile, "%s/virus_mononodo_rzero.csv", risultati);  file8 = fopen(nomefile, "w");
sprintf(nomefile, "%s/virus_mononodo_ctr.csv", risultati);  file9 = fopen(nomefile, "w");
fprintf(file2, "t\tNe0\tPe0\tSe0\tEe0\tIe0\tAe0\tCe0\tRTe0\tRHe0\tRQe0\tRCe0\tGRe0\tGOe0\tDe0\tVe0\t");
fprintf(file2, "t\tNe1\tPe1\tSe1\tEe1\tIe1\tAe1\tCe1\tRTe1\tRHe1\tRQe1\tRCe1\tGRe1\tGOe1\tDe1\tVe1\t");
fprintf(file2, "t\tN\tP\tS\t");
for(vv=0; vv<numvar; vv++) fprintf(file2, "Ev%d\t", vv);
for(vv=0; vv<numvar; vv++) fprintf(file2, "Iv%d\t", vv);
for(vv=0; vv<numvar; vv++) fprintf(file2, "Av%d\t", vv);
for(vv=0; vv<numvar; vv++) fprintf(file2, "Cv%d\t", vv);
fprintf(file2, "C\tRT\tRH\tRQ\tRC\tGR\tGO\tD\tV\tVtot0\tVtot1\n");
for(nn=0; nn<2; nn++)
    {
    fprintf(file2, "%g\t%g\t%g\t%g\t", tt[t1], N[t1][nn], P[t1][nn], S[t1][nn]);
    somma = 0;  for(vv=0; vv<numvar; vv++) somma += E[t1][nn][vv];  fprintf(file2, "%g\t", somma);
    somma = 0;  for(vv=0; vv<numvar; vv++) somma += I[t1][nn][vv];  fprintf(file2, "%g\t", somma);
    somma = 0;  for(vv=0; vv<numvar; vv++) somma += A[t1][nn][vv];  fprintf(file2, "%g\t", somma);
    somma = 0;  for(vv=0; vv<numvar; vv++) somma += C[t1][nn][vv];  fprintf(file2, "%g\t", somma);
    fprintf(file2, "%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t", RT[t1][nn], RH[t1][nn], RQ[t1][nn], RC[t1][nn],
        GR[t1][nn], GO[t1][nn], D[t1][nn], V[t1][nn]);
    }
fprintf(file2, "%g\t%g\t%g\t%g\t", tt[t1], N[t1][0]+N[t1][1], P[t1][0]+P[t1][1], S[t1][0]+S[t1][1]);
fprintf(file2, "%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t",
    E[t1][0][0]+E[t1][1][0], E[t1][0][1]+E[t1][1][1], E[t1][0][2]+E[t1][1][2], E[t1][0][3]+E[t1][1][3],
    I[t1][0][0]+I[t1][1][0], I[t1][0][1]+I[t1][1][1], I[t1][0][2]+I[t1][1][2], I[t1][0][3]+I[t1][1][3],
    A[t1][0][0]+A[t1][1][0], A[t1][0][1]+A[t1][1][1], A[t1][0][2]+A[t1][1][2], A[t1][0][3]+A[t1][1][3],
    C[t1][0][0]+C[t1][1][0], C[t1][0][1]+C[t1][1][1], C[t1][0][2]+C[t1][1][2], C[t1][0][3]+C[t1][1][3]);
fprintf(file2, "%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t0\t0\n",
    C[t1][0][0]+C[t1][1][0]+C[t1][0][1]+C[t1][1][1]+C[t1][0][2]+C[t1][1][2]+C[t1][0][3]+C[t1][1][3],
    RT[t1][0]+RT[t1][1], RH[t1][0]+RH[t1][1], RQ[t1][0]+RQ[t1][1], RC[t1][0]+RC[t1][1],
    GR[t1][0]+GR[t1][1], GO[t1][0]+GO[t1][1], D[t1][0]+D[t1][1], V[t1][0]+V[t1][1]);
fprintf(file6, "t\tgamC\tgamT\tgamH\tdelT\tdelH\n");
fprintf(file8, "t\tRt\n");

double vactot0=0, vactot1=0, vaccinati_totali_0[]={0,0}, vaccinati_totali_1[]={0,0};

// evoluzione temporale
printf("\n\nEVOLUZIONE TEMPORALE [da 0 a %d]:\n", tmax);  fflush(stdout);
qualekalenda = -1;  qualelex = -1;  qualevariante = 1;  qualevac = -1;  senex0 = senex1 = 0;  medicus0 = medicus1 = true;
for(t1=1; t1<=tmax; t1++)
    {
    tt[t1] = t1*dt;  t0 = t1-1;

    if(tt[t1]>=arrivo[qualevariante])
        {
        switch(qualevariante)
            {
            case 1:
                SE[t0][0][1] += 1;
                cout<<"\n>>> ALFA ARRIVATA al tempo "<<tt[t1]<<"\n\n";  fflush(stdout);
                break;
            case 2:
                SE[t0][0][2] += 2;
                cout<<"\n>>> DELTA ARRIVATA al tempo "<<tt[t1]<<"\n\n";  fflush(stdout);
                break;
            case 3:
                SE[t0][0][3] += 1;
                cout<<"\n>>> OMICRON ARRIVATA al tempo "<<tt[t1]<<"\n\n";  fflush(stdout);
                break;
            default:
                cout<<"\n>>> ERRORE SULLA VARIANTE al tempo "<<tt[t1]<<"\n\n";  exit(0);
            }
        qualevariante++;
        }

    if( (tt[t1]<915-inizio_epidemia) || (tt[t1]>=950-inizio_epidemia) )
        { if(tt[t1]>=lex[qualelex+1]) { qualelex++;  mob = mobv[qualelex]; }  }
    else
        { mob = 3.8e12*exp(-0.035*(tt[t1]+inizio_epidemia));  fprintf(nuovofile, "%g\t%g\n", tt[t1], mob); }

    if(tt[t1]>=kalenda[qualekalenda+1])
        {
        qualekalenda++;
        alfa = 0.14;
        if(1.25*alfa>sintom[qualekalenda]) { alfa = sintom[qualekalenda]/1.25; }
        
        gamC = (alfa/(1.0-alfa))*(1.0/sintom[qualekalenda]-1.0);
        gamT = min(0.9, gamTv[qualekalenda]/sintom[qualekalenda]);
        gamH = min(1.0-gamT, gamHv[qualekalenda]/sintom[qualekalenda]);
        //delH = max(0.0, min(0.9, (morti[qualekalenda]/sintom[qualekalenda]-delT*gamT)/gamH));
        delH = max(0.0, min(0.9, (morti[qualekalenda]/(sintom[qualekalenda]*(10.0*gamT+gamH)))));
        delT = max(0.0, min(0.9, 10.0*delH));
        chiQ = min(1.0, max(0.0, (chi-chiT*gamT-chiH*gamH)/(1-gamT-gamH)));
        chiC = chi;
        gammaT[0] = min(1, gamT*(1-chiT)/chi);  gammaT[1] = min(1, gamT*chiT/chi);
        gammaH[0] = min(1-gammaT[0], gamH*(1-chiH)/chi);  gammaH[1] = min(1-gammaT[1], gamH*chiH/chi);

        Tt = round(Ttv[qualekalenda]/dt);  Th = round(Thv[qualekalenda]/dt);
        sqm = 0.1*Tt;  teta = sqm*sqm/Tt;  capp = Tt/teta;
        dismin = max(1, round(Tt-3*sqm));  Lt = round(Tt+5*sqm);  somma = 0;
        for(ii=1; ii<=Lt; ii++) { pdft[ii] = 0; }
        for(ii=dismin; ii<=Lt; ii++) { pdft[ii] = pow(ii,capp-1)*exp(-ii/teta);  somma += pdft[ii]; }
        for(ii=1; ii<=Lt; ii++) { pdft[ii] /= somma; }
        sqm = 0.1*Th;  teta = sqm*sqm/Th;  capp = Th/teta;
        dismin = max(1, round(Th-3*sqm));  Lh = round(Th+5*sqm);  somma = 0;
        for(ii=1; ii<=Lh; ii++) { pdfh[ii] = 0; }
        for(ii=dismin; ii<=Lh; ii++) { pdfh[ii] = pow(ii,capp-1)*exp(-ii/teta);  somma += pdfh[ii]; }
        for(ii=1; ii<=Lh; ii++) { pdfh[ii] /= somma; }
        fprintf(file6, "%g\t%g\t%g\t%g\t%g\t%g\n", tt[t1], gamC, gamT, gamH, delT, delH);  //fflush(file6);
        }

    // distribuzione vaccini prima e seconda dose
    if(medicus0==true && vaccinati_totali_0[0]>=numeromedici)
        {
        for(ii=qualevac; ii<sizeof(pianvac0)/sizeof(pianvac0[0]); ii++) { vac0[ii][1] += vac0[ii][0];  vac0[ii][0] = 0; }
        vaccinati0[0] = vac0[qualevac][0];  vaccinati0[1] = vac0[qualevac][1];  medicus0 = false;
        }
    if(senex0==0 && (1.0-novax)*N[0][1]-giavaccinati0[1]-(vaccinati_totali_0[1]+RT[t0][1]+RH[t0][1]+RQ[t0][1]+RC[t0][1]+GR[t0][1]+D[t0][1])<0.5)
        {
        for(ii=qualevac; ii<sizeof(pianvac0)/sizeof(pianvac0[0]); ii++) { vac0[ii][0] += vac0[ii][1];  vac0[ii][1] = 0; }
        vaccinati0[0] = vac0[qualevac][0];  vaccinati0[1] = vac0[qualevac][1];  senex0 = 1;
        }
    if(senex0==1 && (1.0-novax)*N[0][0]-giavaccinati0[0]-(vaccinati_totali_0[0]+RT[t0][0]+RH[t0][0]+RQ[t0][0]+RC[t0][0]+GR[t0][0]+D[t0][0])<0.5)
        {
        for(ii=qualevac; ii<sizeof(pianvac0)/sizeof(pianvac0[0]); ii++) { vac0[ii][0] = 0;  vac0[ii][1] = 0; }
        vaccinati0[0] = vac0[qualevac][0];  vaccinati0[1] = vac0[qualevac][1];  senex0 = 2;
        }        
    // distribuzione vaccini terza dose 
    if(medicus1==true && vaccinati_totali_1[0]>=numeromedici*efficacia1)
        {
        for(ii=qualevac; ii<sizeof(pianvac1)/sizeof(pianvac1[0]); ii++) { vac1[ii][1] += vac1[ii][0];  vac1[ii][0] = 0; }
        vaccinati1[0] = vac1[qualevac][0];  vaccinati1[1] = vac1[qualevac][1];  medicus1 = false;
        }           
    if(senex1==0 && (1.0-novax)*N[0][1]*efficacia1-(vaccinati_totali_1[1]+RT[t0][1]+RH[t0][1]+RQ[t0][1]+RC[t0][1]+GR[t0][1]+D[t0][1])<0.5)
        {
        for(ii=qualevac; ii<sizeof(pianvac1)/sizeof(pianvac1[0]); ii++) { vac1[ii][0] += vac1[ii][1];  vac1[ii][1] = 0; }
        vaccinati1[0] = vac1[qualevac][0];  vaccinati1[1] = vac1[qualevac][1];  senex1 = 1;
        }
    if(senex1==1 && (1-novax)*N[0][0]*efficacia1-(vaccinati_totali_1[0]+RT[t0][0]+RH[t0][0]+RQ[t0][0]+RC[t0][0]+GR[t0][0]+D[t0][0])<0.5)
        {
        for(ii=qualevac; ii<sizeof(pianvac1)/sizeof(pianvac1[0]); ii++) { vac1[ii][0] = 0;  vac1[ii][1] = 0; }
        vaccinati1[0] = vac1[qualevac][0];  vaccinati1[1] = vac1[qualevac][1];  senex1 = 2;
        }        
    if(tt[t1]>=kalevacc[qualevac+1])
        { 
		qualevac++;
		vaccinati0[0] = vac0[qualevac][0];  vaccinati0[1] = vac0[qualevac][1];
		vaccinati1[0] = vac1[qualevac][0];  vaccinati1[1] = vac1[qualevac][1];
	    }
 
    fprintf(file9, "%g\t%d\t%g\t%g\n", tt[t1]+44, qualevac, vaccinati0[0], vaccinati0[1]);
    somma = 0;  for(nn=0; nn<2; nn++) { for(vv=0; vv<numvar; vv++) somma += E[t0][nn][vv]; }
    printf("t1=%d, tt=%g:  E[t0]=%g\r", t1, tt[t1], somma);  fflush(stdout);

    // flusso S->E
    if(S[t0][0]+S[t0][1]>0)
        {
        //gente = I[t0][nn] * mob[n];  incI = floor[gente];  if [gente - incI > rand[]]  incI++;  end;
        // variante 0 (cinese, ceppo originale)
        gente = (I[t0][0][0]+I[t0][1][0]);  incI = round(gente);
        gente = (A[t0][0][0]+A[t0][1][0]);  incA = round(gente);
        gente = (betaI*incI+betaA*incA)*mob*(S[t0][0]+S[t0][1])/(P[t0][0]+P[t0][1]);
        flussoSE[0][0] = gente*S[t0][0]/(S[t0][0]+S[t0][1]);
        flussoSE[1][0] = gente*S[t0][1]/(S[t0][0]+S[t0][1]);
        // variante 1 (inglese, alfa)
        gente = (I[t0][0][1]+I[t0][1][1]);  incI = round(gente);
        gente = (A[t0][0][1]+A[t0][1][1]);  incA = round(gente);
        gente = (betaI*incI+betaA*incA)*kv[1]*mob*(S[t0][0]+S[t0][1])/(P[t0][0]+P[t0][1]);
        flussoSE[0][1] = gente*S[t0][0]/(S[t0][0]+S[t0][1]);
        flussoSE[1][1] = gente*S[t0][1]/(S[t0][0]+S[t0][1]);
        // variante 2 (indiana, delta)
        gente = (I[t0][0][2]+I[t0][1][2]);  incI = round(gente);
        gente = (A[t0][0][2]+A[t0][1][2]);  incA = round(gente);
        gente = (betaI*incI+betaA*incA)*kv[2]*mob*(S[t0][0]+S[t0][1])/(P[t0][0]+P[t0][1]);
        flussoSE[0][2] = gente*S[t0][0]/(S[t0][0]+S[t0][1]);
        flussoSE[1][2] = gente*S[t0][1]/(S[t0][0]+S[t0][1]);
        // variante 3 (omicron)
        gente = (I[t0][0][3]+I[t0][1][3]);  incI = round(gente);
        gente = (A[t0][0][3]+A[t0][1][3]);  incA = round(gente);
        gente = (betaI*incI+betaA*incA)*kv[3]*mob*(S[t0][0]+S[t0][1])/(P[t0][0]+P[t0][1]);
        flussoSE[0][3] = gente*S[t0][0]/(S[t0][0]+S[t0][1]);
        flussoSE[1][3] = gente*S[t0][1]/(S[t0][0]+S[t0][1]);        

        for(nn=0; nn<2; nn++)
            {
            if(S[t0][nn]>0)
                {
			    somma = 0;  for(vv=0; vv<numvar; vv++) somma += flussoSE[nn][vv];
                if(somma<=S[t0][nn]) { for(vv=0; vv<numvar; vv++) SE[t0][nn][vv] += flussoSE[nn][vv]; }
                else
                    {
                    gente = 0;  for(vv=0; vv<numvar; vv++) gente += flussoSE[nn][vv];
                    if(gente>0) { for(vv=0; vv<numvar; vv++) SE[t0][nn][vv] += flussoSE[nn][vv]*S[t0][nn]/gente; }
                    }
                }
            }
        }

    for(nn=0; nn<2; nn++)
        {

        // bilanciamento dei buffer d'uscita
        for(vv=0; vv<numvar; vv++)
            {
            somma = 0.0;  for(t2=t0+1; t2<=t0+Le; t2++) somma += ge[t2][nn][vv];
            fprintf(file3, "E\t%g\t%d\t%d\t%g\n", t1*dt, nn, vv, E[t0][nn][vv]/somma); //***
            if(somma>0) { somma = E[t0][nn][vv]/somma;  for(t2=t0+1; t2<=t0+Le; t2++) ge[t2][nn][vv] *= somma; }
            else { for(t2=t0+1; t2<=t0+Le; t2++) ge[t2][nn][vv] = E[t0][nn][vv]*pdfe[t2-t1]; }
            somma = 0.0;  for(t2=t0+1; t2<=t0+Li; t2++) somma += gi[t2][nn][vv];
            fprintf(file3, "I\t%g\t%d\t%d\t%g\n", t1*dt, nn, vv, I[t0][nn][vv]/somma); //***
            if(somma>0) { somma = I[t0][nn][vv]/somma;  for(t2=t0+1; t2<=t0+Li; t2++) gi[t2][nn][vv] *= somma; }
            else { for(t2=t0+1; t2<=t0+Li; t2++) gi[t2][nn][vv] = I[t0][nn][vv]*pdfi[t2-t1]; }
            somma = 0.0;  for(t2=t0+1; t2<=t0+La; t2++) somma += ga[t2][nn][vv];
            fprintf(file3, "A\t%g\t%d\t%d\t%g\n", t1*dt, nn, vv, A[t0][nn][vv]/somma); //***
            if(somma>0) { somma = A[t0][nn][vv]/somma;  for(t2=t0+1; t2<=t0+La; t2++) ga[t2][nn][vv] *= somma; }
            else { for(t2=t0+1; t2<=t0+La; t2++) ga[t2][nn][vv] = A[t0][nn][vv]*pdfa[t2-t1]; }
            }
        somma = 0.0;  for(t2=t0+1; t2<=t0+Lt; t2++) somma += gt[t2][nn];
        fprintf(file3, "RT\t%g\t%d\tNaN\t%g\n", t1*dt, nn, RT[t0][nn]/somma); //***
        if(somma>0) { somma = RT[t0][nn]/somma;  for(t2=t0+1; t2<=t0+Lt; t2++) gt[t2][nn] *= somma; }
        else { for(t2=t0+1; t2<=t0+Lt; t2++) gt[t2][nn] = RT[t0][nn]*pdft[t2-t1]; }
        somma = 0.0;  for(t2=t0+1; t2<=t0+Lh; t2++) somma += gh[t2][nn];
        fprintf(file3, "RH\t%g\t%d\tNaN\t%g\n", t1*dt, nn, RH[t0][nn]/somma); //***
        if(somma>0) { somma = RH[t0][nn]/somma;  for(t2=t0+1; t2<=t0+Lh; t2++) gh[t2][nn] *= somma; }
        else { for(t2=t0+1; t2<=t0+Lh; t2++) gh[t2][nn] = RH[t0][nn]*pdfh[t2-t1]; }
        somma = 0.0;  for(t2=t0+1; t2<=t0+Lq; t2++) somma += gq[t2][nn];
        fprintf(file3, "RQ\t%g\t%d\tNaN\t%g\n", t1*dt, nn, RQ[t0][nn]/somma); //***
        if(somma>0) { somma = RQ[t0][nn]/somma;  for(t2=t0+1; t2<=t0+Lq; t2++) gq[t2][nn] *= somma; }
        else { for(t2=t0+1; t2<=t0+Lq; t2++) gq[t2][nn] = RQ[t0][nn]*pdfq[t2-t1]; }     
        somma = 0.0;  for(t2=t0+1; t2<=t0+Lc; t2++) somma += gc[t2][nn];
        fprintf(file3, "RC\t%g\t%d\tNaN\t%g\n", t1*dt, nn, RC[t0][nn]/somma); //***
        if(somma>0) { somma = RC[t0][nn]/somma;  for(t2=t0+1; t2<=t0+Lc; t2++) gc[t2][nn] *= somma; }
        else { for(t2=t0+1; t2<=t0+Lc; t2++) gc[t2][nn] = RC[t0][nn]*pdfc[t2-t1]; }
        somma = 0.0;  for(t2=t0+1; t2<=t0+Lg; t2++) somma += gr[t2][nn];
        fprintf(file3, "GR\t%g\t%d\tNaN\t%g\n", t1*dt, nn, GR[t0][nn]/somma); //***
        if(somma>0) { somma = GR[t0][nn]/somma;  for(t2=t0+1; t2<=t0+Lg; t2++) gr[t2][nn] *= somma; }
        else { for(t2=t0+1; t2<=t0+Lg; t2++) gr[t2][nn]= GR[t0][nn]*pdfg[t2-t1]; }
        somma = 0.0;  for(t2=t0+1; t2<=t0+Lg; t2++) somma += gn[t2][nn];
        fprintf(file3, "GO\t%g\t%d\tNaN\t%g\n", t1*dt, nn, GO[t0][nn]/somma); //***
        if(somma>0) { somma = GO[t0][nn]/somma;  for(t2=t0+1; t2<=t0+Lg; t2++) gn[t2][nn] *= somma; }
        else { for(t2=t0+1; t2<=t0+Lg; t2++) gn[t2][nn] = GO[t0][nn]*pdfg[t2-t1]; }
        somma = 0.0;  for(t2=t0+1; t2<=t0+Lv; t2++) somma += gv[t2][nn];
        fprintf(file3, "V\t%g\t%d\tNaN\t%g\n", t1*dt, nn, V[t0][nn]/somma); //***
        if(somma>0) { somma = V[t0][nn]/somma;  for(t2=t0+1; t2<=t0+Lv; t2++) gv[t2][nn] *= somma; }
        else { for(t2=t0+1; t2<=t0+Lv; t2++) gv[t2][nn] = V[t0][nn]*pdfv[t2-t1]; }

        // flussi di transizione fra compartimenti
        for(vv=0; vv<numvar; vv++)
            {
            // flussi E->I/A
            gente = min(E[t0][nn][vv], ge[t0][nn][vv]);
            EI[t0][nn][vv] += alfa*gente;
            EA[t0][nn][vv] += gente-EI[t0][nn][vv];
            // flussi I->R
            gente = min(I[t0][nn][vv], gi[t0][nn][vv]);
            IRT[vv] = gammaT[nn]*gente;
            IRH[vv] = gammaH[nn]*gente;
            IRQ[vv] = gente-IRT[vv]-IRH[vv];
            // flussi A->RC/GO
            gente = min(A[t0][nn][vv], ga[t0][nn][vv]);
            ARC[vv] = gamC*gente;
            AGO[vv] = gente-ARC[vv];
            }
        // flussi RT->D/RH
        gente = min(RT[t0][nn], gt[t0][nn]);
        RTD = delT*gente;
        RTH = gente-RTD;
        // flussi RH->D/RQ
        gente = min(RH[t0][nn], gh[t0][nn]);
        RHD = delH*gente;
        RHQ = gente-RHD;
        // flusso RQ->GR
        RQG = min(RQ[t0][nn], gq[t0][nn]);
        // flusso RC->GR
        RCG = min(RC[t0][nn], gc[t0][nn]);
        // flusso G->S
        GRS = min(GR[t0][nn], gr[t0][nn]);
        GOS = min(GO[t0][nn], gn[t0][nn]);
        // flusso V->S
        VS = min(V[t0][nn], gv[t0][nn]);
        fprintf(file4, "%d\t%d\t%g\t%g\t%g\n", t1, nn, GRS, GOS, VS);
 
        // popolazioni al tempo t1 e relativi flussi netti d'uscita
        S[t1][nn] = S[t0][nn]+GOS;  for(vv=0; vv<numvar; vv++) S[t1][nn] -= SE[t0][nn][vv];
        for(vv=0; vv<numvar; vv++)
            {
            E[t1][nn][vv] = E[t0][nn][vv] + SE[t0][nn][vv] - (EI[t0][nn][vv]+EA[t0][nn][vv]);
            for(t2=t1+1; t2<=t1+Le; t2++)
               { ge[t2][nn][vv] += SE[t0][nn][vv]*pdfe[t2-t1];  if(ge[t2][nn][vv]<0.0) ge[t2][nn][vv] = 0.0; }
            I[t1][nn][vv] = I[t0][nn][vv] + EI[t0][nn][vv] - (IRT[vv]+IRH[vv]+IRQ[vv]);
            for(t2=t1+1; t2<=t1+Li; t2++)
                { gi[t2][nn][vv] += EI[t0][nn][vv]*pdfi[t2-t1];  if(gi[t2][nn][vv]<0.0) gi[t2][nn][vv] = 0.0; }
            A[t1][nn][vv] = A[t0][nn][vv] + EA[t0][nn][vv] - (ARC[vv]+AGO[vv]);
            for(t2=t1+1; t2<=t1+La; t2++)
                { ga[t2][nn][vv] += EA[t0][nn][vv]*pdfa[t2-t1];  if(ga[t2][nn][vv]<0.0) ga[t2][nn][vv] = 0.0; }
            }
        RT[t1][nn] = RT[t0][nn]-(RTH+RTD);  for(vv=0; vv<numvar; vv++) RT[t1][nn] += IRT[vv];
		gente = 0;  for(vv=0; vv<numvar; vv++) gente += IRT[vv];
        for(t2=t1+1; t2<=t1+Lt; t2++) { gt[t2][nn] += gente*pdft[t2-t1];  if(gt[t2][nn]<0.0) gt[t2][nn] = 0.0; }
        RH[t1][nn] = RH[t0][nn] + RTH - (RHQ+RHD);  for(vv=0; vv<numvar; vv++) RH[t1][nn] += IRH[vv];
		gente = RTH;  for(vv=0; vv<numvar; vv++) gente += IRH[vv];
        for(t2=t1+1; t2<=t1+Lh; t2++) { gh[t2][nn] += gente*pdfh[t2-t1];  if(gh[t2][nn]<0.0) gh[t2][nn] = 0.0; }
        RQ[t1][nn] = RQ[t0][nn] + RHQ - RQG;  for(vv=0; vv<numvar; vv++) RQ[t1][nn] += IRQ[vv];
		gente = RHQ;  for(vv=0; vv<numvar; vv++) gente += IRQ[vv];
        for(t2=t1+1; t2<=t1+Lq; t2++) { gq[t2][nn] += gente*pdfq[t2-t1];  if(gq[t2][nn]<0.0) gq[t2][nn] = 0.0; }
        RC[t1][nn] = RC[t0][nn] - RCG;  for(vv=0; vv<numvar; vv++) RC[t1][nn] += ARC[vv];
		gente = 0;  for(vv=0; vv<numvar; vv++) gente += ARC[vv];
        for(t2=t1+1; t2<=t1+Lc; t2++)
            { gc[t2][nn] += gente*pdfc[t2-t1];  if(gc[t2][nn]<0.0) gc[t2][nn] = 0.0; }
        for(vv=0; vv<numvar; vv++) C[t1][nn][vv] = C[t0][nn][vv] + IRT[vv]+IRH[vv]+IRQ[vv]+ARC[vv];
        GR[t1][nn] = GR[t0][nn] + (RQG+RCG) - GRS;
        for(t2=t1+1; t2<=t1+Lg; t2++) { gr[t2][nn] += (RQG+RCG)*pdfg[t2-t1];  if(gr[t2][nn]<0.0) gr[t2][nn] = 0.0; }
        GO[t1][nn] = GO[t0][nn] - GOS;  for(vv=0; vv<numvar; vv++) GO[t1][nn] += AGO[vv];
        for(t2=t1+1; t2<=t1+Lg; t2++)
            { gn[t2][nn] += (GO[t1][nn]-GO[t0][nn]+GOS)*pdfg[t2-t1];  if(gn[t2][nn]<0.0) gn[t2][nn] = 0.0; }
        D[t1][nn] = D[t0][nn] + (RTD+RHD);          
      
        // vaccinazioni
        vactot0 = 0;  vaccinabili0 = S[t1][nn]+GO[t1][nn]-giavaccinati0[nn];
        for(vv=0; vv<numvar; vv++) vaccinabili0 += E[t1][nn][vv]+I[t1][nn][vv]+A[t1][nn][vv];
        if(vaccinabili0>0)
            { 
			gente = min(S[t1][nn], vaccinati0[nn]*(S[t1][nn]-giavaccinati0[nn])/vaccinabili0);  S[t1][nn] -= gente;  vactot0 += gente;
            for(vv=0; vv<numvar; vv++)
                {
			    gente = min(E[t1][nn][vv], vaccinati0[nn]*E[t1][nn][vv]/vaccinabili0);  vactot0 += gente;  E[t1][nn][vv] -= gente;
			    for(t2=t1+1; t2<=t1+Le; t2++) { ge[t2][nn][vv] -= gente*pdfe[t2-t1];  if(ge[t2][nn][vv]<0.0) ge[t2][nn][vv] = 0.0; }
                gente = min(I[t1][nn][vv], vaccinati0[nn]*I[t1][nn][vv]/vaccinabili0);  vactot0 += gente;  I[t1][nn][vv] -= gente;
			    for(t2=t1+1; t2<=t1+Li; t2++) { gi[t2][nn][vv] -= gente*pdfi[t2-t1];  if(gi[t2][nn][vv]<0.0) gi[t2][nn][vv] = 0.0; }
                gente = min(A[t1][nn][vv], vaccinati0[nn]*A[t1][nn][vv]/vaccinabili0);  vactot0 += gente;  A[t1][nn][vv] -= gente;
                for(t2=t1+1; t2<=t1+La; t2++) { ga[t2][nn][vv] -= gente*pdfa[t2-t1];  if(ga[t2][nn][vv]<0.0) ga[t2][nn][vv] = 0.0; }
		    	}
            gente = min(GO[t1][nn], vaccinati0[nn]*GO[t1][nn]/vaccinabili0);  vactot0 += gente;  GO[t1][nn] -= gente;
            for(t2=t1+1; t2<=t1+Lg; t2++) { gn[t2][nn] -= gente*pdfg[t2-t1];  if(gn[t2][nn]<0.0) gn[t2][nn] = 0.0; }
		    }
		residuo = vaccinati1[nn];  vactot1 = 0;
        if(VS<=residuo) { vactot1 += VS;  residuo -= VS; }
        else { vactot1 += residuo;  S[t1][nn] += VS-residuo;  giavaccinati0[nn] += VS-residuo;  residuo = 0; }
        if(GRS<=residuo) { vactot1 += GRS;  residuo -= GRS; }
        else { vactot1 += residuo;  S[t1][nn] += GRS-residuo;  giavaccinati0[nn] += GRS-residuo;  residuo = 0; }
        if(residuo>0)
            { 
		    vaccinabili1 = S[t1][nn]+GO[t1][nn]+GR[t1][nn];
            for(vv=0; vv<numvar; vv++) vaccinabili1 += E[t1][nn][vv]+I[t1][nn][vv]+A[t1][nn][vv];
            if(vaccinabili1>0)
                {
			    residuo /= vaccinabili1;
			    gente = min(S[t1][nn], residuo*S[t1][nn]);  S[t1][nn] -= gente;  vactot1 += gente;
                for(vv=0; vv<numvar; vv++)
                    {
			        gente = min(E[t1][nn][vv], residuo*E[t1][nn][vv]);  vactot1 += gente;  E[t1][nn][vv] -= gente;
			        for(t2=t1+1; t2<=t1+Le; t2++)
                        { ge[t2][nn][vv] -= gente*pdfe[t2-t1];  if(ge[t2][nn][vv]<0.0) ge[t2][nn][vv] = 0.0; }
                    gente = min(I[t1][nn][vv], residuo*I[t1][nn][vv]);  vactot1 += gente;  I[t1][nn][vv] -= gente;
                    for(t2=t1+1; t2<=t1+Li; t2++)
                        { gi[t2][nn][vv] -= gente*pdfi[t2-t1];  if(gi[t2][nn][vv]<0.0) gi[t2][nn][vv] = 0.0; }
                    gente = min(A[t1][nn][vv], residuo*A[t1][nn][vv]);  vactot1 += gente;  A[t1][nn][vv] -= gente;
                    for(t2=t1+1; t2<=t1+La; t2++)
                        { ga[t2][nn][vv] -= gente*pdfa[t2-t1];  if(ga[t2][nn][vv]<0.0) ga[t2][nn][vv] = 0.0; }
		    	    }
                gente = min(GR[t1][nn], residuo*GR[t1][nn]);  GR[t1][nn] -= gente;  vactot1 += gente;
                for(t2=t1+1; t2<=t1+Lg; t2++)
                    { gr[t2][nn] -= gente*pdfg[t2-t1];  if(gr[t2][nn]<0.0) gn[t2][nn] = 0.0; }
                gente = min(GO[t1][nn], residuo*GO[t1][nn]);  GO[t1][nn] -= gente;  vactot1 += gente;
                for(t2=t1+1; t2<=t1+Lg; t2++)
                    { gn[t2][nn] -= gente*pdfg[t2-t1];  if(gn[t2][nn]<0.0) gn[t2][nn] = 0.0; }
				}
		    }        
        V[t1][nn] = max( 0.0 , V[t0][nn] + vactot0+vactot1 - VS );
        for(t2=t1+1; t2<=t1+Lv; t2++)
            { gv[t2][nn] += (vactot0+vactot1)*pdfv[t2-t1];  if(gv[t2][nn]<0.0) gv[t2][nn] = 0.0; }
        vaccinati_totali_0[nn] += vactot0;  vaccinati_totali_1[nn] += vactot1;    

        P[t1][nn] = S[t1][nn] + GR[t1][nn] + GO[t1][nn] + V[t1][nn];
        for(vv=0; vv<numvar; vv++) P[t1][nn] += E[t1][nn][vv] + I[t1][nn][vv] + A[t1][nn][vv];
        N[t1][nn] = P[t1][nn] + RT[t1][nn] + RH[t1][nn] + RQ[t1][nn] + RC[t1][nn] + D[t1][nn];

        fprintf(file2, "%g\t%g\t%g\t%g\t", tt[t1], N[t1][nn], P[t1][nn], S[t1][nn]);
        somma = 0;  for(vv=0; vv<numvar; vv++) somma += E[t1][nn][vv];  fprintf(file2, "%g\t", somma);
        somma = 0;  for(vv=0; vv<numvar; vv++) somma += I[t1][nn][vv];  fprintf(file2, "%g\t", somma);
        somma = 0;  for(vv=0; vv<numvar; vv++) somma += A[t1][nn][vv];  fprintf(file2, "%g\t", somma);
        somma = 0;  for(vv=0; vv<numvar; vv++) somma += C[t1][nn][vv];  fprintf(file2, "%g\t", somma);
        fprintf(file2, "%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t", RT[t1][nn], RH[t1][nn], RQ[t1][nn], RC[t1][nn],
            GR[t1][nn], GO[t1][nn], D[t1][nn], V[t1][nn]);
        }
 
    fprintf(file2, "%g\t%g\t%g\t%g\t", tt[t1], N[t1][0]+N[t1][1], P[t1][0]+P[t1][1], S[t1][0]+S[t1][1]);
    fprintf(file2, "%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t",
        E[t1][0][0]+E[t1][1][0], E[t1][0][1]+E[t1][1][1], E[t1][0][2]+E[t1][1][2], E[t1][0][3]+E[t1][1][3],
        I[t1][0][0]+I[t1][1][0], I[t1][0][1]+I[t1][1][1], I[t1][0][2]+I[t1][1][2], I[t1][0][3]+I[t1][1][3],
        A[t1][0][0]+A[t1][1][0], A[t1][0][1]+A[t1][1][1], A[t1][0][2]+A[t1][1][2], A[t1][0][3]+A[t1][1][3],
        C[t1][0][0]+C[t1][1][0], C[t1][0][1]+C[t1][1][1], C[t1][0][2]+C[t1][1][2], C[t1][0][3]+C[t1][1][3]);
    fprintf(file2, "%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n",
        C[t1][0][0]+C[t1][1][0]+C[t1][0][1]+C[t1][1][1]+C[t1][0][2]+C[t1][1][2]+C[t1][0][3]+C[t1][1][3],
        RT[t1][0]+RT[t1][1], RH[t1][0]+RH[t1][1], RQ[t1][0]+RQ[t1][1], RC[t1][0]+RC[t1][1],
        GR[t1][0]+GR[t1][1], GO[t1][0]+GO[t1][1], D[t1][0]+D[t1][1], V[t1][0]+V[t1][1],
        vaccinati_totali_0[0]+vaccinati_totali_0[1], vaccinati_totali_1[0]+vaccinati_totali_1[1]);

    gente = 0.0;  erret = 0.0;
    for(vv=0; vv<numvar; vv++)
        {
        gente += I[t1][0][vv]+I[t1][1][vv]+A[t1][0][vv]+A[t1][1][vv];
        erret += (I[t1][0][vv]+I[t1][1][vv]+A[t1][0][vv]+A[t1][1][vv])*kv[vv];
        }   
    if(gente>0.0) { erret *= mob*betaI*Ti*(S[t1][0]+S[t1][1])/((P[t1][0]+P[t1][1])*gente); }
    else { erret = kv[qualevariante-1]*mob*betaI*Ti*(S[t1][0]+S[t1][1])/((P[t1][0]+P[t1][1])); }
    fprintf(file8, "%g\t%g\n", tt[t1], erret);

    }

printf("\nevoluzione terminata.\n\n");

fclose(file2);  fclose(file3);  fclose(file4);  fclose(file6);  fclose(file8);

}
