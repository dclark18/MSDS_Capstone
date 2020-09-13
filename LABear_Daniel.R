#This file will perform a step analysis for male and female black bears radiocollared in Louisiana
#Joe Clark- 9/1/2020

setwd ("D:/Google Drive/JCLARK/BEAR/Louisiana/Joe-GPS collar data/Corridor analysis UTM")
getwd()


#Read in the raw data (time, date, lat, long, steplength...)
attributes <- read.table("allbear9att.txt", header=TRUE, sep=",")
summary(attributes)

#create 1 time categories (0-4 hr) variable and assign it to the data frame
timecats.4hr=cut(attributes$TIMEINTVL,breaks=c(200,14760),labels=
	c("0-4hr"),include.lowest=T,right=F)
attributesa=cbind(attributes,timecats.4hr)
summary(attributesa)
str(attributesa) 

#define time stamp 
library(lubridate)
attributesa$UTCDateTime<-as.POSIXct(strptime(as.character(attributesa$GPS_Fix_Time),"%m/%d/%Y %H:%M:%S"))
attributesa<-attributesa%>%filter(!is.na(`UTCDateTime`)) 
all(complete.cases(attributesa$UTCDateTime))

##Now get rid of observations <200 seconds apart and >14760 (4.1) hrs apart (i.e., NAs)
attributes.4hr <- subset(attributesa, TIMEINTVL > 200 & TIMEINTVL < 14760) 
summary(attributes.4hr)

#Get rid of observations with very short steps (resting) or very long ones (error) and also the initial
#step which would not have a turning angle (-999)
attributes.4hr.long <- subset(attributes.4hr, STEPLENGTH > 100 & TURNANGLE > -999 & STEPLENGTH < 10000) 
summary(attributes.4hr.long)


#Make separate dataframes for all MALE bears
all_males <- subset(attributes.4hr.long, Sex==1) #Males only
unique(all_males$Bear_ID)
#I am sure a for loop could be written for this.....
M1<-all_males[all_males$Bear_ID=="1",]
M2<-all_males[all_males$Bear_ID=="2",]
M3<-all_males[all_males$Bear_ID=="3",]
M4<-all_males[all_males$Bear_ID=="4",]
M5<-all_males[all_males$Bear_ID=="5",]
M6<-all_males[all_males$Bear_ID=="6",]
M8<-all_males[all_males$Bear_ID=="8",]
M9<-all_males[all_males$Bear_ID=="9",]
M10<-all_males[all_males$Bear_ID=="10",]
M12<-all_males[all_males$Bear_ID=="12",]
M13<-all_males[all_males$Bear_ID=="13",]
M14<-all_males[all_males$Bear_ID=="14",]
M16<-all_males[all_males$Bear_ID=="16",]
M17<-all_males[all_males$Bear_ID=="17",]
M19<-all_males[all_males$Bear_ID=="19",]
M21<-all_males[all_males$Bear_ID=="21",]
M69<-all_males[all_males$Bear_ID=="69",]
M79<-all_males[all_males$Bear_ID=="79",]
M80<-all_males[all_males$Bear_ID=="80",]
M84<-all_males[all_males$Bear_ID=="84",]
M711<-all_males[all_males$Bear_ID=="711",]
M790<-all_males[all_males$Bear_ID=="790",]
M831<-all_males[all_males$Bear_ID=="831",]
str(M1)
#create tracks for all bears
#********************* DOUBLE CHECK COORDINATE TRANSFORMATIONS!!! *********************************
library(amt)
M1_tk<-mk_track(M1, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M1_tk)
M2_tk<-mk_track(M2, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M2_tk)
M3_tk<-mk_track(M3, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M3_tk)
M4_tk<-mk_track(M4, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M4_tk)
M3_tk<-mk_track(M3, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M3_tk)
M5_tk<-mk_track(M5, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M5_tk)
M6_tk<-mk_track(M6, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M6_tk)
M8_tk<-mk_track(M8, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M8_tk)
M9_tk<-mk_track(M9, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M9_tk)
M10_tk<-mk_track(M10, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M10_tk)
M12_tk<-mk_track(M12, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M12_tk)
M13_tk<-mk_track(M13, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M13_tk)
M14_tk<-mk_track(M14, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M14_tk)
M16_tk<-mk_track(M16, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M16_tk)
M17_tk<-mk_track(M17, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M17_tk)
M19_tk<-mk_track(M19, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M19_tk)
M21_tk<-mk_track(M21, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M21_tk)
M69_tk<-mk_track(M69, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M69_tk)
M79_tk<-mk_track(M79, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M79_tk)
M80_tk<-mk_track(M80, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M80_tk)
M84_tk<-mk_track(M84, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M84_tk)
M711_tk<-mk_track(M711, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M711_tk)
M790_tk<-mk_track(M790, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M790_tk)
M831_tk<-mk_track(M831, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(M831_tk)

#This summarizes the sampling rate to give you an idea of how many observations there were
#and hthe time intervals between locations, e.g.,:
summarize_sampling_rate(M1_tk)  
summarize_sampling_rate(M2_tk)  
summarize_sampling_rate(M3_tk)  
summarize_sampling_rate(M4_tk)  
summarize_sampling_rate(M5_tk)  

#keep only bursts (subsets of the track with constant sampling rate, within the specified tolerance)
#with at least three relocations, the minimum required to calculate a turn angle
#In this example the locations had to be 4 hrs apart (plus or munus 2 hrs) and there had to be
#at least 3 points in a row with these attributes (so that a turning angle could be calculated)
#You can change this to whatever you want but be aware of the limitations of the data (see sampling
#rate above for examples)
M1_steps<-track_resample(M1_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M1_steps <- M1_steps %>% random_steps(n = 9)  

M2_steps<-track_resample(M2_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M2_steps <- M2_steps %>% random_steps(n = 9)  

M3_steps<-track_resample(M3_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M3_steps <- M3_steps %>% random_steps(n = 9)  

M4_steps<-track_resample(M4_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M4_steps <- M4_steps %>% random_steps(n = 9)  

M5_steps<-track_resample(M5_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M5_steps <- M5_steps %>% random_steps(n = 9)  

M6_steps<-track_resample(M6_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M6_steps <- M6_steps %>% random_steps(n = 9)  

M8_steps<-track_resample(M8_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M8_steps <- M8_steps %>% random_steps(n = 9)  

M9_steps<-track_resample(M9_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M9_steps <- M9_steps %>% random_steps(n = 9)  

M10_steps<-track_resample(M10_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M10_steps <- M10_steps %>% random_steps(n = 9)  

M12_steps<-track_resample(M12_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M12_steps <- M12_steps %>% random_steps(n = 9)  

M13_steps<-track_resample(M13_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M13_steps <- M13_steps %>% random_steps(n = 9)  

M14_steps<-track_resample(M14_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M14_steps <- M14_steps %>% random_steps(n = 9)  

M16_steps<-track_resample(M16_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M16_steps <- M16_steps %>% random_steps(n = 9)  

M17_steps<-track_resample(M17_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M17_steps <- M17_steps %>% random_steps(n = 9)  

M19_steps<-track_resample(M19_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M19_steps <- M19_steps %>% random_steps(n = 9)  

M21_steps<-track_resample(M21_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M21_steps <- M21_steps %>% random_steps(n = 9)  

M69_steps<-track_resample(M69_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M69_steps <- M69_steps %>% random_steps(n = 9)  

M79_steps<-track_resample(M79_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M79_steps <- M79_steps %>% random_steps(n = 9)  

M80_steps<-track_resample(M80_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M80_steps <- M80_steps %>% random_steps(n = 9)  

M84_steps<-track_resample(M84_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M84_steps <- M84_steps %>% random_steps(n = 9)  

M711_steps<-track_resample(M711_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M711_steps <- M711_steps %>% random_steps(n = 9)  

M790_steps<-track_resample(M790_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M790_steps <- M790_steps %>% random_steps(n = 9)  

M831_steps<-track_resample(M831_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
M831_steps <- M831_steps %>% random_steps(n = 9)  

##################################################################

#Load in and name rasters for plotting purposes
#You won't have these so just refer to the RData file I gave you
library(raster)

#Distance to nearest agriculture
disag<-raster("E:/ARCGIS DATA/Louisiana/stepselection/dist_ag_2")
names(disag)<-"disag"
plot(disag)
#Distance to nearest forest
disfor<-raster("E:/ARCGIS DATA/Louisiana/stepselection/dist_for_4")
names(disfor)<-"disfor"
#Distance to nearest natural (forest, shrubs, wetlands, etc.)
disnat<-raster("E:/ARCGIS DATA/Louisiana/stepselection/dist_nat_4")
names(disnat)<-"disnat"
#Distance to nearest road
disrd<-raster("E:/ARCGIS DATA/Louisiana/stepselection/dist_rds_3")
names(disrd)<-"disrd"
#Distance to nearest water
diswat<-raster("E:/ARCGIS DATA/Louisiana/stepselection/dist_water")
names(diswat)<-"diswat"
#Density of natural cover types (percent - 0 to 1)
natden<-raster("E:/ARCGIS DATA/Louisiana/stepselection/nat_den_5")
names(natden)<-"natden"
#Density of forest cover types (percent - 0 to 1)
forden<-raster("E:/ARCGIS DATA/Louisiana/stepselection/for_den_3")
names(forden)<-"forden"
#Density of roads (percent - 0 to 1)
rdden<-raster("E:/ARCGIS DATA/Louisiana/stepselection/rd_dens")
names(rdden)<-"rdden"
plot(rdden) 
####################################################


#Create a function that extracts covariates
covariate_extraction<-function(data){
  data <- data %>% extract_covariates(disag, where="end") 
  data <- data %>% extract_covariates(disfor, where="end") 
  data <- data %>% extract_covariates(disnat, where="end") 
  data <- data %>% extract_covariates(disrd, where="end") 
  data <- data %>% extract_covariates(diswat, where="end") 
  data <- data %>% extract_covariates(natden, where="end") 
  data <- data %>% extract_covariates(forden, where="end") 
  data <- data %>% extract_covariates(rdden, where="end") 
  bear<<-data
}

covariate_extraction(M1_steps)
M1_steps<-bear
M1_steps$ID<-"1"
summary(M1_steps)
covariate_extraction(M1_steps)
M2_steps<-bear
M2_steps$ID<-"2"
summary(M2_steps)
covariate_extraction(M3_steps)
M3_steps<-bear
M3_steps$ID<-"3"
summary(M3_steps)
covariate_extraction(M4_steps)
M4_steps<-bear
M4_steps$ID<-"4"
summary(M4_steps)
covariate_extraction(M5_steps)
M5_steps<-bear
M5_steps$ID<-"5"
summary(M5_steps)
covariate_extraction(M6_steps)
M6_steps<-bear
M6_steps$ID<-"6"
summary(M6_steps)
covariate_extraction(M8_steps)
M8_steps<-bear
M8_steps$ID<-"8"
summary(M8_steps)
covariate_extraction(M9_steps)
M9_steps<-bear
M9_steps$ID<-"9"
summary(M9_steps)
covariate_extraction(M10_steps)
M10_steps<-bear
M10_steps$ID<-"10"
summary(M10_steps)
covariate_extraction(M12_steps)
M12_steps<-bear
M12_steps$ID<-"12"
summary(M12_steps)
covariate_extraction(M13_steps)
M13_steps<-bear
M13_steps$ID<-"13"
summary(M13_steps)
covariate_extraction(M14_steps)
M14_steps<-bear
M14_steps$ID<-"14"
summary(M14_steps)
covariate_extraction(M16_steps)
M16_steps<-bear
M16_steps$ID<-"16"
summary(M16_steps)
covariate_extraction(M17_steps)
M17_steps<-bear
M17_steps$ID<-"17"
summary(M17_steps)
covariate_extraction(M19_steps)
M19_steps<-bear
M19_steps$ID<-"19"
summary(M19_steps)
covariate_extraction(M21_steps)
M21_steps<-bear
M21_steps$ID<-"21"
summary(M21_steps)
covariate_extraction(M69_steps)
M69_steps<-bear
M69_steps$ID<-"69"
summary(M69_steps)
covariate_extraction(M79_steps)
M79_steps<-bear
M79_steps$ID<-"79"
summary(M79_steps)
covariate_extraction(M80_steps)
M80_steps<-bear
M80_steps$ID<-"80"
summary(M80_steps)
covariate_extraction(M84_steps)
M84_steps<-bear
M84_steps$ID<-"84"
summary(M84_steps)
covariate_extraction(M711_steps)
M711_steps<-bear
M711_steps$ID<-"711"
summary(M711_steps)
covariate_extraction(M790_steps)
M790_steps<-bear
M790_steps$ID<-"790"
summary(M790_steps)
covariate_extraction(M831_steps)
M831_steps<-bear
M831_steps$ID<-"831"
summary(M831_steps)

malebears<-rbind(M1_steps,M2_steps,M3_steps,M4_steps,M5_steps,M6_steps,M8_steps,M9_steps,
                      M10_steps,M12_steps,M13_steps,M14_steps,M16_steps,M17_steps,M19_steps,M21_steps,
                      M69_steps,M79_steps,M80_steps,M84_steps,M711_steps,M790_steps,M831_steps) #

#Create strata for the case-control pairing
malebears$case_
STRATUM=rep(1:5468, each = 10, len = 54680)
malebears=cbind(malebears,STRATUM)
summary(malebears)

#####################
#Make separate dataframes for all FEMALE bears
#####################
all_females <- subset(attributes.4hr.long, Sex==2) #Males only
unique(all_females$Bear_ID)
#I am sure a for loop could be written for this.....
F7<-all_females[all_females$Bear_ID=="7",]
F18<-all_females[all_females$Bear_ID=="18",]
F20<-all_females[all_females$Bear_ID=="20",]
F71<-all_females[all_females$Bear_ID=="71",]
F375<-all_females[all_females$Bear_ID=="375",]
F713<-all_females[all_females$Bear_ID=="713",]
F714<-all_females[all_females$Bear_ID=="714",]

#create tracks for all bears
#********************* DOUBLE CHECK COORDINATE TRANSFORMATIONS!!! *********************************
library(amt)
F7_tk<-mk_track(F7, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(F7_tk)
F18_tk<-mk_track(F18, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(F18_tk)
F20_tk<-mk_track(F20, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(F20_tk)
F71_tk<-mk_track(F71, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(F71_tk)
F375_tk<-mk_track(F375, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(F375_tk)
F713_tk<-mk_track(F713, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(F713_tk)
F714_tk<-mk_track(F714, .x=GPS_Longitude, .y=GPS_Latitude, .t=UTCDateTime, ID=Bear_ID,  crs = sp::CRS("+init=epsg:26915"))
plot(F714_tk)

#This summarizes the sampling rate to give you an idea of how many observations there were
#and hthe time intervals between locations, e.g.,:
summarize_sampling_rate(F7_tk)  

#keep only bursts (subsets of the track with constant sampling rate, within the specified tolerance)
#with at least three relocations, the minimum required to calculate a turn angle
#In this example the locations had to be 4 hrs apart (plus or munus 2 hrs) and there had to be
#at least 3 points in a row with these attributes (so that a turning angle could be calculated)
#You can change this to whatever you want but be aware of the limitations of the data (see sampling
#rate above for examples)
F7_steps<-track_resample(F7_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
F7_steps <- F7_steps %>% random_steps(n = 9)  

F18_steps<-track_resample(F18_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
F18_steps <- F18_steps %>% random_steps(n = 9)  

F20_steps<-track_resample(F20_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
F20_steps <- F20_steps %>% random_steps(n = 9)  

F71_steps<-track_resample(F71_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
F71_steps <-F71_steps %>% random_steps(n = 9)  

F375_steps<-track_resample(F375_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
F375_steps <- F375_steps %>% random_steps(n = 9)  

F713_steps<-track_resample(F713_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
F713_steps <- F713_steps %>% random_steps(n = 9)  

F714_steps<-track_resample(F714_tk,rate=hours(4),tolerance = hours(2))%>%
  filter_min_n_burst(min_n=3)%>%  steps_by_burst()%>%
  time_of_day(include.crepuscule=TRUE)
#create 9 random steps for each true step
F714_steps <- F714_steps %>% random_steps(n = 9)  

covariate_extraction(F7_steps)
F7_steps<-bear
F7_steps$ID<-"7"
summary(F7_steps)
covariate_extraction(F18_steps)
F18_steps<-bear
F18_steps$ID<-"18"
summary(F18_steps)
covariate_extraction(F20_steps)
F20_steps<-bear
F20_steps$ID<-"20"
summary(F20_steps)
covariate_extraction(F71_steps)
F71_steps<-bear
F71_steps$ID<-"71"
summary(F71_steps)
covariate_extraction(F375_steps)
F375_steps<-bear
F375_steps$ID<-"375"
summary(F375_steps)
covariate_extraction(F713_steps)
F713_steps<-bear
F713_steps$ID<-"713"
summary(F713_steps)
covariate_extraction(F714_steps)
F714_steps<-bear
F714_steps$ID<-"714"
summary(F714_steps)

femalebears<-rbind(F7_steps,F18_steps,F20_steps,F71_steps,F375_steps,F713_steps,F714_steps) #
summary(femalebears)

#Create strata for the case-control pairing
femalebears$case_
STRATUM=rep(1:721, each = 10, len = 7210)
femalebears=cbind(femalebears,STRATUM)
summary(femalebears)

#Now for some logistic regression
library (survival)
female.ssf.model1 <- clogit(case_ ~ disnat + disag + disrd 
                            + strata(STRATUM), data=femalebears, method='approximate')




#This stuff below is junk but it might be useful 
########################
#Step 1. Individual main effects in a Mixed effects context 
#This is probably the preferred way to do it because it is based on both a random slope
#and random intercept   This is probably overly complicated and I was getting an error message
library(glmmTMB)
summary(malebears$case_)
#A few male models - Conditional logistic regression-mixed effects
#disag 
M.1<-glmmTMB(case_ ~  disag + (1|step_id_) + (0 + disag|ID),
             family=poisson, data=malebears, doFit=FALSE)
M.1$parameters$theta[1]=log(1e3)
M.1$parameters$theta
M.1$mapArg = list(theta=factor(c(NA,1:1))) # list 1:x where x is # of random effects
MM.1<-glmmTMB:::fitTMB(M.1)
summary(MM.1) #disag not significant
AIC(MM.1) #    

#disfor 
M.2<-glmmTMB(case_ ~  disfor + (1|step_id_) + (0 + disfor|ID),
             family=poisson, data=malebears, doFit=FALSE)
M.2$parameters$theta[1]=log(1e3)
M.2$parameters$theta
M.2$mapArg = list(theta=factor(c(NA,1:1))) # list 1:x where x is # of random effects
MM.2<-glmmTMB:::fitTMB(M.2)
summary(MM.2) #disfor significant 8.25e-09 ***
AIC(MM.2) #    

female.ssf.model1 <- clogit(OBSERVED ~ disnatEND + disagEND + distrdsEND +  disnatEND*distrdsEND + disnatEND*disagEND + cluster(CLUSTER)
		 + strata(STRATUM), data=femaleclean3, method='approximate')
female.Global.2 <- clogit(OBSERVED ~ disagEND + disnatEND + distrdsEND + 
	disnatEND*distrdsEND + disnatEND*disagEND + cluster(CLUSTER) + strata(STRATUM), data=femalecleannoNA, 
	method='approximate', na.action = na.fail)
female.Global.3 <- clogit(OBSERVED ~ disagEND + disnatEND + distrdsEND + distwatEND +
	disnatEND*distrdsEND + disnatEND*disagEND + cluster(CLUSTER) + strata(STRATUM), data=femalecleannoNA, 
	method='approximate', na.action = na.fail)
ftemp <- clogit(OBSERVED ~ distwatEND + cluster(CLUSTER)
		 + strata(STRATUM), data=femaleclean3, method='approximate')
female.ssf.tmp <- clogit(OBSERVED ~ disnatLWM + disagEND + distrdsEND +  cluster(CLUSTER)
		 + strata(STRATUM), data=femaleclean3, method='approximate')

female.ssf.tmp2 <- clogit(OBSERVED ~ disnatLWM + disagLWM + distrdsLWM +  disnatLWM*distrdsLWM + disnatLWM*disagLWM + cluster(CLUSTER)
		 + strata(STRATUM), data=femaleclean3, method='approximate')
female.Global.2 <- clogit(OBSERVED ~
summary(female.ssf.model1)
extractAIC(female.ssf.model1)
extractAIC(female.ssf.tmp2)
plot(femaleclean2$NatnegLWM, femaleclean2$OBSERVED)
plot(femaleclean2$disnatEND, femaleclean2$OBSERVED)
summary(femaleclean2)
extractAIC(female.ssf.model1)
summary(female.ssf.model1)
print(summary(female.ssf.model)$coef[,1:2] )
cat(paste("Model prediction", exp(female.ssf.model$coef[2]), "\n"))

logitmodf=(exp(predict(female.ssf.model)))/(1+exp(predict(female.ssf.model)))
plot(femaleclean2$NatnegLWM, logitmodf, xlab="Distance to Natural", 
	ylab="Probability of Use")
plot(femaleclean2$NatnegLWM, predict(female.ssf.model), xlab="Distance to Natural", 
	ylab="Logit of Probability of Use")
plot(femaleclean2$disagMIN, logitmodf,  xlab="disagMIN", 
	ylab="Probability of Use")
plot(femaleclean2$disagMIN, predict(female.ssf.model), xlab="disagMIN", 
	ylab="Logit of Probability of Use")
plot(femaleclean2$distrdsMAX, logitmodf,  xlab="distrdsMAX", 
	ylab="Probability of Use")
plot(femaleclean2$distrdsMAX, predict(female.ssf.model), xlab="distrdsMAX", 
	ylab="Logit of Probability of Use")
# standardize beta coefficients for comparison
sdxf=cbind(sd(femaleclean2$disnatEND), sd(femaleclean2$disagEND), sd(femaleclean2$distrdsEND), sd(femaleclean2$disnatEND*femaleclean2$distrdsEND), sd(femaleclean2$disnatEND*femaleclean2$disagEND))
stdxybetaf=(coef(female.ssf.model1)*sdxf)/sd(predict(female.ssf.model1))
stdxbetaf=(coef(female.ssf.model1)*sdxf)



#model for season
femaleclean4 <- read.csv("./stepselection/femaleclean4.csv", header=TRUE, sep=",")
attach(femaleclean4)
femaleclean4$season[month > 11] <- 0
 femaleclean4$season[month > 6 & month <= 11] <- 1
 femaleclean4$season[month <= 6] <- 0
 detach(femaleclean4) 
summary(femaleclean4)
female.season <- clogit(OBSERVED ~ disnatEND + disagEND + distrdsEND + season + disnatEND*distrdsEND + disnatEND*disagEND + disagEND*season +  cluster(CLUSTER)
		 + strata(STRATUM), data=femaleclean4, method='approximate')
female.seasontemp <- clogit(OBSERVED ~ disnatEND + disagEND + distrdsEND +  disnatEND*distrdsEND + disnatEND*disagEND + disagEND*season +   cluster(CLUSTER)
		 + strata(STRATUM), data=femaleclean4, method='approximate')
#extractAIC(female.temp)
#summary(female.season)
#summary(femaledat4)
(xtabs(~disagEND+season, data=femaleclean4))
)
#####to estimate robust SEs######
femaleclean2$resid<-residuals(female.ssf.model3,type="deviance")
tempresidtotF<-by(femaleclean2,femaleclean2$STRATUM,function(x) sum(x$resid))
residtotF<-data.frame("STRATUM"=as.numeric(names(tempresidtotF)),"resid"=as.numeric(unlist(tempresidtotF)))
residtotF<-merge(femaleclean2[femaleclean2$OBSERVED==1,c("FID","Bear_ID","SAMPLEID","STRATUM")],residtotF)
summary(residtotF)
temp.lmeF<-try(lme(resid~1,random=~1|Bear_ID,data=residtotF,correlation=corAR1(form=~FID|Bear_ID)))
ACF(temp.lmeF, maxLag=40)
plot(ACF(temp.lmeF, maxLag=40), alpha = 0.05)
#(this shows that the model in not autocorrelated)
#Pick the value where ACF<0.05
#Write the data to an excel file and go in and number the clusters 1 and -1 by 10
write.csv(femaleclean2,file="D:/Google Drive/JCLARK/BEAR/Louisiana/stepselection/femaleclean3.csv") 
femaleclean3 <- read.csv("femaleclean3.csv", header=TRUE, sep=",")

####Then rerun the same model using the cluster option: 
female.ssf.model.clust <- clogit(OBSERVED ~ disnatEND + disagMIN + distrdsEND  + disnatEND*distrdsEND + disnatEND*disagMIN + cluster(CLUSTER)
		 + strata(STRATUM), data=femaleclean3, method='approximate')
summary(female.ssf.model.clust)
str(female.ssf.model.clust)


#a random effect model for females
library(coxme)

#attach(femaleclean3)
# femaledat3$season[month > 11] <- 0
 #femaledat3$season[month > 6 & month <= 11] <- 1
# femaledat3$season[month <= 6] <- 0
# detach(femaledat3) 
#str(femaledat3)
frand=coxme(Surv(rep(1, 26467L), OBSERVED)~disnatEND + disagEND + distrdsEND + disnatEND*distrdsEND + disnatEND*disagEND +
              (disnatEND|Bear_ID) + strata(STRATUM), data = femaleclean4)
summary(frand)
fixed.effects(frand)
random.effects(frand)

fnorand=coxph(Surv(rep(1, 26467L), OBSERVED)~disnatEND + disagEND + distrdsEND +  disnatEND*distrdsEND + disnatEND*disagEND +
		strata(STRATUM), data = femaleclean4)
#LR tests won't work with coxme models
anova(frand,fnorand)

#could not figure out how to calculate robust SEs for mixed effects

####THE MALE MODEL 
male.ssf.model1 <- clogit(OBSERVED ~ disnatEND + disagEND +  distrdsEND +  disnatEND*disagEND + 
    + I(disnatEND^2) + I(distrdsEND^2) + strata(STRATUM)+ cluster(CLUSTER), data=maleclean4, method='approximate')
extractAIC(male.ssf.model1)

male.ssf.model2 <- clogit(OBSERVED ~ disnatEND + disagEND +  distrdsEND +  
    + I(disnatEND^2) + I(distrdsEND^2) + strata(STRATUM)+ cluster(CLUSTER), data=maleclean4, method='approximate')
extractAIC(male.ssf.model2)

male.ssf.temp <- clogit(OBSERVED ~ disnatEND + disagEND +    disnatEND*disagEND + 
    + I(disnatEND^2)  + strata(STRATUM)+ cluster(CLUSTER), data=maleclean3, method='approximate')

extractAIC(male.ssf.temp)
summary(maleclean3)

male.ssf.temp <- clogit(OBSERVED ~ disnatEND + disagEND +  distrdsEND +  disnatEND*disagEND + 
    + I(disnatEND^2) + I(distrdsEND^2) + strata(STRATUM), data=maleclean2, method='approximate')
extractAIC(male.ssf.temp)
summary(male.ssf.temp)
male.ssf.temp3 <- clogit(OBSERVED ~ disnatEND + disagEND +  rddenLWM +  disnatEND*disagEND + 
    + I(disnatEND^2) + I(rddenLWM^2) + strata(STRATUM), data=maleclean2, method='approximate')
extractAIC(male.ssf.temp3)
male.ssf.temp2 <- clogit(OBSERVED ~ NatnegLWM + disagEND +  distrdsEND +  
    + I(NatnegLWM^2)  + strata(STRATUM), data=maleclean2, method='approximate')
extractAIC(male.ssf.temp2)
summary(male.ssf.temp2)
summary(male.ssf.model)

# standardize beta coefficients for comparison
sdxm=cbind(sd(maleclean2$disnatEND), sd(maleclean2$disagEND), sd(maleclean2$distrdsEND), sd((maleclean2$disnatEND)^2), sd((maleclean2$distrdsEND)^2), sd(maleclean2$disnatEND*maleclean2$disagEND))
stdxybetam=(coef(male.ssf.model)*sdxm)/sd(predict(male.ssf.model))
stdxbetam=(coef(male.ssf.model)*sdxm)
sd((maleclean3$disnatEND)^2)





#####to estimate robust SEs######
maleclean2$resid<-residuals(male.ssf.temp,type="deviance")
tempresidtotM<-by(maleclean2,maleclean2$STRATUM,function(x) sum(x$resid))
residtotM<-data.frame("STRATUM"=as.numeric(names(tempresidtotM)),"resid"=as.numeric(unlist(tempresidtotM)))
residtotM<-merge(maleclean2[maleclean2$OBSERVED==1,c("FID","Bear_ID","SAMPLEID","STRATUM")],residtotM)
summary(residtotM)
temp.lmeM<-try(lme(resid~1,random=~1|Bear_ID,data=residtotM,correlation=corAR1(form=~FID|Bear_ID)))
ACF(temp.lmeM, maxLag=40)
plot(ACF(temp.lmeM, maxLag=40), alpha = 0.05)

#Pick the value where ACF<0.05 (looks like 24)
#Write the data to an excel file and go in and number the clusters 1 and -1 by 10
write.csv(maleclean2,file="D:/Google Drive/JCLARK/BEAR/Louisiana/stepselection/maleclean3.csv") 
##number clusters and change the sign every other cluster in excel.
maleclean3 <- read.csv("maleclean3.csv", header=TRUE, sep=",")

####Then rerun the same model using the cluster option: 
male.ssf.model <- clogit(OBSERVED ~ disnatEND + disagEND +  distrdsEND +  disnatEND*disagEND + 
    + I(disnatEND^2) + I(distrdsEND^2) + strata(STRATUM)+ cluster(CLUSTER), data=maleclean3, method='approximate')

#model for season
maleclean4 <- read.csv("D:/Google Drive/JCLARK/BEAR/Louisiana/stepselection/stepselection/maleclean4.csv", header=TRUE, sep=",")
attach(maleclean4)
maleclean4$season[month > 11] <- 0
 maleclean4$season[month > 6 & month <= 11] <- 1
 maleclean4$season[month <= 6] <- 0
 detach(maleclean4) 
summary(maleclean4)
male.season <- clogit(OBSERVED ~ disnatEND + disagEND +  distrdsEND + season + disnatEND*disagEND + 
    + I(disnatEND^2) + I(distrdsEND^2) + disagEND*season  + strata(STRATUM)+ cluster(CLUSTER), data=maleclean4, method='approximate')



#a random effect model for males
library(coxme)
attach(maledat4)
 maledat4$season[month > 11] <- 0
 maledat4$season[month > 6 & month <= 11] <- 1
 maledat4$season[month <= 6] <- 0
 detach(maledat4) 
summary(maledat4)
mrand=coxme(Surv(rep(1, 167058L), OBSERVED)~disnatEND + disagEND +  distrdsEND +  disnatEND*disagEND +
        I(disnatEND^2) + I(distrdsEND^2) + (disnatEND|Bear_ID) + strata(STRATUM), data = maleclean3)

mnorand=coxph(Surv(rep(1, 167058L), OBSERVED)~disnatEND + disagEND +  distrdsEND +  disnatEND*disagEND + 
        I(disnatEND^2) + I(distrdsEND^2) + strata(STRATUM), data = maleclean3)
anova(mrand,mnorand)
fixed.effects(mrand)
random.effects(mrand)




