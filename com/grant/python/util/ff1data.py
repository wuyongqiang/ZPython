'''
Created on Mar 19, 2018

@author: grant
'''

import csv;


def readDataSets(train_dir,priceIncrease=1.02,useDailyPriceChange=False):    
    return TradingData(train_dir,priceIncrease,useDailyPriceChange);

class TradingData:

    def __init__(self, dataFilePath, priceIncrease=1.02, useDailyPriceChange=False):
        
        self.dataFilePath = dataFilePath;
        self.currentPosition = 0;
        self.priceIncrease = priceIncrease;
        self.useDailyPriceChange = useDailyPriceChange;
        
        self.dataRows = [];
        with open(self.dataFilePath, 'rb') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.dataRows.append(row)
            
            self.dataRows.reverse()    
            f.close()
    
    def rowCount(self):
        return len(self.dataRows)         
    
    def nextBatch(self, batchSize):
        data = [];
        label = [];
        
        if self.currentPosition < 0:
            return [],[];
        
        batch = 0;
        latestClosingPrice = 0.000000001;
        while batch < batchSize:
            
            dataSize = 0;
            rowNumber =  0;
            maxPriceOfNextWeek = 0;            
            firstDayPrice = 0.000000001;
            for row in self.dataRows:
                rowNumber += 1;
                #skip data that have been read
                if rowNumber <= self.currentPosition:                
                    continue
                #endif 
                
                #skip bad data
                if float(row['Close']) < 0.000000001 or float(row['Open']) < 0.000000001 or row['Shares on Issue'] == '':
                    self.currentPosition += 1
                    continue
                
                dataSize += 1;
                
                if dataSize <= 20:
                    if firstDayPrice < 0.000000002:
                        firstDayPrice = float(row['Open']);
                    if latestClosingPrice < 0.000000002:
                        latestClosingPrice = firstDayPrice;
                    
                    open_price = float(row['Open'].replace(',',''));  
                    low_price = float(row['Low'].replace(',','')); 
                    high_price = float(row['High'].replace(',','')); 
                    close_price = float(row['Close'].replace(',',''));
                    volume = float(row['Volume'].replace(',',''));  
                    shares = float(row['Shares on Issue'].replace(',','')); 
                    if not self.useDailyPriceChange:
                        data.extend([open_price/firstDayPrice, high_price/firstDayPrice, low_price/firstDayPrice, close_price/firstDayPrice, volume/shares])
                    else:
                        data.extend([open_price/latestClosingPrice-1, high_price/latestClosingPrice-1, low_price/latestClosingPrice-1, close_price/latestClosingPrice-1, volume/shares])
                        latestClosingPrice = close_price;  
                        
                elif dataSize <= 25: #data of next week
                    price = float(row['High']);
                    if price > maxPriceOfNextWeek:
                        maxPriceOfNextWeek = price;
                else:
                    break;
                #endif
            #end for
            if (maxPriceOfNextWeek/latestClosingPrice) > self.priceIncrease: 
                label.append( 1 );
            else:
                label.append( 0.0 );
                  
            if dataSize < 25:
                self.currentPosition = -1;
                return [],[];
            #end if                
            batch += 1;
            self.currentPosition += 1;    
        #end while    
    
        return data, label;
            
    def nextLstmBatch(self, batchSize):
        data = [];
        label = [];
        
        if self.currentPosition < 0:
            return [],[];
        
        batch = 0;
        firstDayPrice = 0.000000001;
        
        while batch < batchSize:
            
            dataSize = 0;
            rowNumber =  0;
            maxPriceOfNextWeek = 0;
            latestClosingPrice = 0.000000001;
            
            for row in self.dataRows:
                rowNumber += 1;
                #skip data that have been read
                if rowNumber <= self.currentPosition:                
                    continue
                #endif 
                
                #skip bad data
                if float(row['Close']) < 0.000000001 or float(row['Open']) < 0.000000001 or row['Shares on Issue'] == '':
                    self.currentPosition += 1
                    continue
                
                dataSize += 1;
                
                if dataSize <= 20:
                    if firstDayPrice < 0.000000002:
                        firstDayPrice = float(row['Open']);
                    latestClosingPrice = float(row['Close']);  
                    open_price = float(row['Open'].replace(',',''));  
                    low_price = float(row['Low'].replace(',','')); 
                    high_price = float(row['High'].replace(',','')); 
                    close_price = float(row['Close'].replace(',',''));
                    volume = float(row['Volume'].replace(',',''));  
                    shares = float(row['Shares on Issue'].replace(',','')); 
                    if dataSize == 20:
                        data.extend([open_price/firstDayPrice, high_price/firstDayPrice, low_price/firstDayPrice, close_price/firstDayPrice, volume/shares])
                
                elif dataSize <= 25: #data of next week
                    price = float(row['High']);
                    if price > maxPriceOfNextWeek:
                        maxPriceOfNextWeek = price;
                else:
                    break;
                #endif
            #end for
            if (maxPriceOfNextWeek/latestClosingPrice) > self.priceIncrease: 
                label.append( 1 );
            else:
                label.append( 0.0 );
                  
            if dataSize < 25:
                self.currentPosition = -1;
                return [],[];
            #end if                
            batch += 1;
            self.currentPosition += 1;    
        #end while    
    
        return data, label;


    