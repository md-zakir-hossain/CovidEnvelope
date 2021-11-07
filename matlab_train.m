#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 21:36:48 2021

@author: yan181
"""

clc;
clear;
clear all;
close all;
fclose('all');
tic
%% Input data file

metadata = jsondecode(fileread('metadata.json'));

for i = 1:size(metadata,1)
    FileName{i,1} = metadata{i}.filename; % read filenames
    
    COVID19{i,1} = metadata{i}.covid19; % covid = 1; non-covid = 0;
    
    if isfield(metadata{i},'verified')
        Verified{i,1} = metadata{i}.verified; % Is covid-test verified? 
    else 
        Verified{i,1} = 'NaN';
    end
end

Data = [FileName COVID19 Verified]; % All metadata (1 = true, 0 = false, [] = null)

%% Detecting cough signal
DataPath = 'C:\Users\basuk\OneDrive - UTS\Covid-19\dataset-main\raw';
nSample = numel(FileName); 
b2=fopen('extracted_features_area.txt','w');
b3=fopen('extracted_features_condition.txt','w');
b4=fopen('extracted_features_mean.txt','w');

for j = 1:nSample
    
%% Read signal

    [AudioSignal,Fs] = audioread([DataPath,'\',Data{j,1}]);
    [~,col1] = size(AudioSignal);

%% Find correct audio signal

    if col1 == 1
        AudioSignal_1 = AudioSignal;
    elseif sum(var(AudioSignal(:,2)))>sum(var(AudioSignal(:,1)))
        AudioSignal_1 = AudioSignal(:,2);  
    else
        AudioSignal_1 = AudioSignal(:,1);
    end
    b1=movmean(AudioSignal_1,3);
    [up,lo]=envelope(b1,30,'peak');
    upper=up(1:length(up));
    lower=lo(1:length(lo));
    b1=abs(upper-lower);
    a=sum(b1);
    if a<5000
        b=1;
    else
        b=0;
    end
    m=mean(b1);
    fprintf(b2,'%2f, ',a);
    fprintf(b3,'%2f, ',b);
    fprintf(b4,'%2f, ',m);
    
end
c=load('extracted_features_area.txt');
c1=c(:);
d=load('extracted_features_condition.txt');
d1=d(:);
e=load('extracted_features_mean.txt');
e1=e(:);
% Data1 = [FileName COVID19 Verified d];
toc