#-*- coding: utf-8 -*-
'''
Created on Jan 20, 2018

@author: grant
'''
#from __future__ import print_function;

from util.TestClasses import *;
import argparse;
import csv;
import tensorflow as tf;

from util.ff1data import readDataSets;

def addElement(x,y): 
    return -x+y

def sumRange(seq):
    return reduce(addElement, seq, 0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    parser.add_argument('--arg2',type=int)
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS.data_dir, FLAGS.arg2);
    
    print(unparsed);
  
    print ("abc");
    
    myList=[1,2,3,4,5,6];
    
    print (myList);
    
    print catchADog();
    
    #print (sys.argv());

    print ("好的");
     
    print (u'\u9f99\u98de\u51e4\u821e');
    
    print (range(-10, -130, -60));  
    
    print (range(11));  
    
    print (sumRange([1,2,3,4,5,6,7]));

    print ([(x, x**2) for x in range(6)]);
    
    vec = [[1,2,3], [4,5,6], [7,8,9]];
    print ([e for e in vec]);
    print ([num**2 for elem in vec for num in elem]);
    
    seta = set('abracadabra');
    print (seta);
    
    setb = set(['abc', 'def']);
    print (setb);
    
    
    for i in reversed(xrange(0,12,1)):
        print (i);
 
    cat = Cat("kitty");
    cat.add_trick("meow!!")
    dog = Dog("puppy");
    dog.add_trick("ruff")
    
    print(cat.get_name(),"abc");
    
    print( dog.get_name() + " "  + str(dog.get_tricks()))
    
    i = 0;
    
    print " 10 mod 3 =" + str( 10/3) 
    
    dataSet = readDataSets('/home/grant/Downloads/phd_data/AMC2017-2018.csv');
    
    print dataSet.nextBatch(3);
    
    print dataSet.nextBatch(3);
    
    with open('/home/grant/Downloads/phd_data/AMC2005-2007.csv', 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ##print (row['Open'], row['High'], row['Volume']);
            print (row['Volume'], row['Shares on Issue'], float(row['Volume'])/float(row['Shares on Issue'])*100, row['High']);
            i = i+1;
            if (i > 2 ) : 
                break;
    
    