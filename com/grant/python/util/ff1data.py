'''
Created on Mar 19, 2018

@author: grant
'''

import csv;


def readDataSets(train_dir):    
    return TradingData(train_dir);

class TradingData:

    def __init__(self, dataFilePath):
        self.dataFilePath = dataFilePath;
        self.currentPosition = 0;
        
        
    def nextBatch(self, batchSize):
        data = [];
        label = [];
        
        if self.currentPosition < 0:
            return [],[];
        
        batch = 0;
        while batch < batchSize:
            with open(self.dataFilePath, 'rb') as f:
                reader = csv.DictReader(f)
                dataSize = 0;
                rowNumber =  0;
                maxPriceOfNextWeek = 0;
                latestClosingPrice = 0.000000001;
                for row in reader:
                    rowNumber += 1;
                    #skip data that have been read
                    if rowNumber <= self.currentPosition:                
                        continue;
                    #endif 
                    
                    dataSize += 1;
                    
                    if dataSize <= 5: #data of next week
                        price = float(row['High']);
                        if price > maxPriceOfNextWeek:
                            maxPriceOfNextWeek = price;
                    elif dataSize <= 25:
                        if latestClosingPrice < 0.000000002:
                            latestClosingPrice = float(row['Close']);
                        #latestClosingPrice = 1;  
                        open_price = float(row['Open'].replace(',',''));  
                        low_price = float(row['Low'].replace(',','')); 
                        high_price = float(row['High'].replace(',','')); 
                        close_price = float(row['Close'].replace(',',''));
                        volume = float(row['Volume'].replace(',',''));  
                        shares = float(row['Shares on Issue'].replace(',','')); 
                        
                        data.extend([open_price/latestClosingPrice, high_price/latestClosingPrice, low_price/latestClosingPrice, close_price/latestClosingPrice, volume/shares])
                    else:
                        break;
                    #endif
                #end for
                if (maxPriceOfNextWeek/latestClosingPrice) > 1.05: 
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


    