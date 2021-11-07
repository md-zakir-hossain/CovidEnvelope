#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 21:37:34 2021

@author: yan181
"""

clc;
clear;
clear all;
close all;
fclose('all');

[AudioSignal, Fs] = audioread('positive2.mp3');
[~, col1] = size(AudioSignal);

%% Find correct audio signal 

    if col1 == 1
        AudioSignal_1 = AudioSignal;
    elseif sum(var(AudioSignal(:,2)))>sum(var(AudioSignal(:,1)))
        AudioSignal_1 = AudioSignal(:,2);  
    else
        AudioSignal_1 = AudioSignal(:,1);
    end

%% Plot correct audio signal   

b=AudioSignal_1;
subplot 411
plot(b)
xlabel('Samples');
ylabel('Amplitude');
axis([0 7000 -1 1]);
b1=movmean(b,3);
subplot 412
plot(b1)
xlabel('Samples');
ylabel('Amplitude');
axis([0 7000 -1 1]);

[up,lo]=envelope(b1,30,'peak');
upper=up(1:length(up));
t=1:length(upper);
subplot 413
plot(t,upper,'r');
hold on
lower=lo(1:length(lo));
t=1:length(lower);
plot(t,lower,'blu');
xlabel('Samples');
ylabel('Amplitude');
axis([0 7000 -1 1]);

b=abs(upper-lower);
t=1:length(b);
subplot 414
plot(t,b);
xlabel('Samples');
ylabel('Amplitude');
axis([0 7000 0 2]);
hold on
sum=sum(b)
average=mean(b)