//+------------------------------------------------------------------+
//|                                                 Precios.mq5       |
//|                                   Copyright 2022, John K Quezada. |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, John Quezada."
#property version   "1.00"
#property strict
#property script_show_inputs

input string Date="2000.01.01 00:00";   //Fecha Inicio
input string DateOut="2021.12.31 23:00"; //Fecha Final
input int History=0;    
input int RsiPeriod=14;

input int Kperiod=5;
input int Dperiod=3;
input int Slowing=3;

input int Ma_periodLento=150;
input int Ma_periodRapido=300;

double inB[22];
string Date1;


int HandleInpuNet5Min;
int HandleInpuNet5Max;

int HandleOutputNet5Min;
int HandleOutput5Max;

double DibMin1_1[];
double DibMax1_1 [];
int DibMin1_1Handle;
int DibMax1_1Handle;

double Stochastic0[];
double Stochastic1[];
double RSI_Close[];

double IMA_CloseRapido01[]; //minutos 150
double IMA_CloseLento01[]; //minutos 300
double IMA_CloseRapido02[]; //15minutos 150
double IMA_CloseLento02[]; //15minutos 300
double IMA_CloseRapido03[]; //1Hora 150
double IMA_CloseLento03[]; //1Hora 300

void OnStart()
  {
//---
   int k=iBars(NULL,PERIOD_M5)-1; //Retorna el número de barras en la historia según el símbolo y el periodo correspondientes.

   DibMin1_1Handle=iCustom(NULL,PERIOD_M5,"DibMin5Min",History); // Busca el punto mas bajo del dia
   CopyBuffer(DibMin1_1Handle,0,0,k,DibMin1_1);
   ArraySetAsSeries(DibMin1_1,true);

   DibMax1_1Handle=iCustom(NULL,PERIOD_M5,"DibMin5Max",History); // Busca el punto mas alto del dia
   CopyBuffer(DibMax1_1Handle,0,0,k,DibMax1_1);
   ArraySetAsSeries(DibMax1_1,true);

   HandleInpuNet5Min=FileOpen(Symbol()+"InputNet5Min.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   HandleInpuNet5Max=FileOpen(Symbol()+"InputNet5Max.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   FileSeek(HandleInpuNet5Min,0,SEEK_END);
   FileSeek(HandleInpuNet5Max,0,SEEK_END);
 
   ////////////////////////////////////////////  
   HandleOutputNet5Min=FileOpen(Symbol()+"OutputNet5Min.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   HandleOutput5Max=FileOpen(Symbol()+"OutputNet5Max.csv",FILE_CSV|FILE_WRITE|FILE_SHARE_READ|FILE_ANSI|FILE_COMMON,";");
   FileSeek(HandleOutputNet5Min,0,SEEK_END);
   FileSeek(HandleOutput5Max,0,SEEK_END);
   ///////////////////////////////////////////////
   int Stochastic_handle=iStochastic(NULL,PERIOD_M5,Kperiod,Dperiod,Slowing,MODE_SMA,STO_LOWHIGH);
   CopyBuffer(Stochastic_handle,0,0,k,Stochastic0);
   CopyBuffer(Stochastic_handle,1,0,k,Stochastic1);
   ArraySetAsSeries(Stochastic0,true);
   ArraySetAsSeries(Stochastic1,true);
   
   int RSI_Close_handle=iRSI(NULL,PERIOD_M5,RsiPeriod,PRICE_CLOSE);
   CopyBuffer(RSI_Close_handle,0,0,k,RSI_Close);
   ArraySetAsSeries(RSI_Close,true);
   
   // cinco minutos
   int Ema_rapido_01=iMA(NULL,PERIOD_M5,Ma_periodLento,0,MODE_LWMA,PRICE_CLOSE);
   int Ema_lento_01=iMA(NULL,PERIOD_M5,Ma_periodRapido,0,MODE_LWMA,PRICE_CLOSE);
   CopyBuffer(Ema_rapido_01,0,0,k,IMA_CloseRapido01);
   ArraySetAsSeries(IMA_CloseRapido01,true);
   
   CopyBuffer(Ema_lento_01,0,0,k,IMA_CloseLento01);
   ArraySetAsSeries(IMA_CloseLento01,true);
   // 15 minutos
   int quince=0;
   int Ema_rapido_02=iMA(NULL,PERIOD_M15,Ma_periodLento,0,MODE_LWMA,PRICE_CLOSE);
   int Ema_lento_02=iMA(NULL,PERIOD_M15,Ma_periodRapido,0,MODE_LWMA,PRICE_CLOSE);
   CopyBuffer(Ema_rapido_02,0,0,k,IMA_CloseRapido02);
   ArraySetAsSeries(IMA_CloseRapido02,true);
   
   CopyBuffer(Ema_lento_02,0,0,k,IMA_CloseLento02);
   ArraySetAsSeries(IMA_CloseLento02,true);
   // una hora
   int hora=0;
   int Ema_rapido_03=iMA(NULL,PERIOD_H1,Ma_periodLento,0,MODE_LWMA,PRICE_CLOSE);
   int Ema_lento_03=iMA(NULL,PERIOD_H1,Ma_periodRapido,0,MODE_LWMA,PRICE_CLOSE);
   CopyBuffer(Ema_rapido_03,0,0,k,IMA_CloseRapido03);
   ArraySetAsSeries(IMA_CloseRapido03,true);
   
   CopyBuffer(Ema_lento_03,0,0,k,IMA_CloseLento03);
   ArraySetAsSeries(IMA_CloseLento03,true);
   ////////////////
   
   if(HandleInpuNet5Min>0)
     {
      Alert("Escribiendo el archivo InputNet1Min");
      Alert("Escribiendo el archivo OutputNet1Min");
      
      for(int i=iBars(NULL,PERIOD_M5)-1; i>=0; i--)
        {
         Date1=TimeToString(iTime(NULL,PERIOD_M5,i));
         if(DateOut>=Date1 && Date<=Date1)
           {
            if((DibMin1_1[i]==-1 && DibMin1_1[i+1]==1 && DibMax1_1[i]==1) || (DibMin1_1[i]==1 && DibMax1_1[i]==1))
              {
              inB[0]=(iOpen(Symbol(),PERIOD_M5,i)-iLow(Symbol(),PERIOD_M5,i))*10000;
              inB[1]=(iHigh(Symbol(),PERIOD_M5,i)-iOpen(Symbol(),PERIOD_M5,i))*10000;
              inB[2]=(iHigh(Symbol(),PERIOD_M5,i)-iLow(Symbol(),PERIOD_M5,i))*10000;
              inB[3]=(iOpen(Symbol(),PERIOD_M5,i)-iClose(Symbol(),PERIOD_M5,i))*10000;
              inB[4]=(iHigh(Symbol(),PERIOD_M5,i+1)-iLow(Symbol(),PERIOD_M5,i))*10000;
              inB[5]=(iHigh(Symbol(),PERIOD_M5,i+1)-iOpen(Symbol(),PERIOD_M5,i))*10000;
              inB[6]=(iHigh(Symbol(),PERIOD_M5,i+1)-iLow(Symbol(),PERIOD_M5,i+1))*10000;
              //////////////////////////////////////
              inB[7]=IMA_CloseRapido01[i]-IMA_CloseLento01[i];
              inB[9]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseLento01[i])*10000;
              inB[10]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseRapido01[i])*10000;
              inB[11]=(iClose(Symbol(),PERIOD_M5,i+1)-IMA_CloseLento01[i+1])*10000;
              inB[12]=(iClose(Symbol(),PERIOD_M5,i+1)-IMA_CloseRapido01[i+1])*10000;
              ///////////////////
              quince=iBarShift(NULL,PERIOD_M15,iTime(NULL,PERIOD_M5,i));
              inB[13]=(IMA_CloseRapido02[quince]-IMA_CloseLento02[quince])*10000;
              inB[14]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseLento01[quince])*10000;
              inB[15]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseRapido01[quince])*10000;
              //////////////////////////////////////////////////////////////////
              hora =iBarShift(NULL,PERIOD_H1,iTime(NULL,PERIOD_M5,i));  
              inB[16]=(IMA_CloseRapido03[hora]-IMA_CloseLento03[hora])*10000;
              inB[17]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseLento03[hora])*10000;
              inB[18]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseRapido03[hora])*10000;
              ////////////////////////////////////////////////////////////////////
              inB[19]=RSI_Close[i];
              inB[20]=RSI_Close[i+1];
              inB[21]=RSI_Close[i+2];        
              FileWrite(HandleInpuNet5Min,inB[0],inB[1],inB[2],inB[3],inB[4],inB[5],inB[6],inB[7],inB[8],inB[9],inB[10],inB[11],inB[12],inB[13],inB[14],inB[15],inB[16],inB[17],inB[18],inB[19],inB[20],inB[21]);
              FileWrite(HandleOutputNet5Min,
              (iClose(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_M5,i)))-iOpen(NULL,PERIOD_M5,i)));
              }
           }
        }

      FileClose(HandleInpuNet5Min);
      FileClose(HandleOutputNet5Min);
     }
//------------------------------------------------------------------------------------------------------------------------------------------------

   if(HandleInpuNet5Max>0)
     {
      Alert("Escribiendo el Archivo InputNet1Max");
      Alert("Escribiendo el Archivo OutputNet1Max");
      
      for(int i=iBars(NULL,PERIOD_M5)-1; i>=0; i--)
        {

         Date1=TimeToString(iTime(NULL,PERIOD_M5,i));

         if(DateOut>=Date1 && Date<=Date1)
           {
            if((DibMax1_1[i]==-1 && DibMax1_1[i+1]==1 && DibMin1_1[i]==1)|| (DibMin1_1[i]==1 && DibMax1_1[i]==1))
              {
               inB[0]=(iOpen(Symbol(),PERIOD_M5,i)-iLow(Symbol(),PERIOD_M5,i))*10000;
              inB[1]=(iHigh(Symbol(),PERIOD_M5,i)-iOpen(Symbol(),PERIOD_M5,i))*10000;
              inB[2]=(iHigh(Symbol(),PERIOD_M5,i)-iLow(Symbol(),PERIOD_M5,i))*10000;
              inB[3]=(iOpen(Symbol(),PERIOD_M5,i)-iClose(Symbol(),PERIOD_M5,i))*10000;
              inB[4]=(iHigh(Symbol(),PERIOD_M5,i+1)-iLow(Symbol(),PERIOD_M5,i))*10000;
              inB[5]=(iHigh(Symbol(),PERIOD_M5,i+1)-iOpen(Symbol(),PERIOD_M5,i))*10000;
              inB[6]=(iHigh(Symbol(),PERIOD_M5,i+1)-iLow(Symbol(),PERIOD_M5,i+1))*10000;
              //////////////////////////////////////
              inB[7]=(IMA_CloseRapido01[i]-IMA_CloseLento01[i])*10000;
              inB[9]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseLento01[i])*10000;
              inB[10]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseRapido01[i])*10000;
              inB[11]=(iClose(Symbol(),PERIOD_M5,i+1)-IMA_CloseLento01[i+1])*10000;
              inB[12]=(iClose(Symbol(),PERIOD_M5,i+1)-IMA_CloseRapido01[i+1])*10000;
              ///////////////////
              quince=iBarShift(NULL,PERIOD_M15,iTime(NULL,PERIOD_M5,i));
              inB[13]=(IMA_CloseRapido02[quince]-IMA_CloseLento02[quince])*10000;
              inB[14]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseLento01[quince])*10000;
              inB[15]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseRapido01[quince])*10000;
              //////////////////////////////////////////////////////////////////
              hora =iBarShift(NULL,PERIOD_H1,iTime(NULL,PERIOD_M5,i));  
              inB[16]=(IMA_CloseRapido03[hora]-IMA_CloseLento03[hora])*10000;
              inB[17]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseLento03[hora])*10000;
              inB[18]=(iClose(Symbol(),PERIOD_M5,i)-IMA_CloseRapido03[hora])*10000;
                 
              inB[19]=RSI_Close[i];
              inB[20]=RSI_Close[i+1];
              inB[21]=RSI_Close[i+2];            
              FileWrite(HandleInpuNet5Max,inB[0],inB[1],inB[2],inB[3],inB[4],inB[5],inB[6],inB[7],inB[8],inB[9],inB[10],inB[11],inB[12],inB[13],inB[14],inB[15],inB[16],inB[17],inB[18],inB[19],inB[20],inB[21]);
             
              FileWrite(HandleOutput5Max,
              (iClose(NULL,PERIOD_M5,i)-iClose(NULL,PERIOD_D1,iBarShift(NULL,PERIOD_D1,iTime(NULL,PERIOD_M5,i))))*10000);
                         
              }
           }
        }

      FileClose(HandleInpuNet5Max);
      FileClose(HandleOutput5Max);
     }
   Alert("Archivos Escritos");
  }
//+------------------------------------------------------------------+
