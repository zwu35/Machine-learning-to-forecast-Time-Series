#The Time Series we use.
d1 <- read.csv("NDDUJN.csv",header = TRUE)
d2 <- read.csv("NDDUE15X.csv",header = TRUE)
d3 <- read.csv("A.csv",header = TRUE)
d4 <- read.csv("NDDUNA.csv",header = TRUE)
d5 <- read.csv("NDDUPXJ.csv",header = TRUE)
d6 <- read.csv("NDDUUK.csv",header = TRUE)
d7 <- read.csv("NDDUWXUS.csv",header = TRUE)
d8 <- read.csv("RU10GRTR.csv",header = TRUE)
d9 <- read.csv("RU20GRTR.csv",header = TRUE)
d10 <- read.csv("S5COND.csv",header = TRUE)
d11 <- read.csv("S5ENRS.csv",header = TRUE)


#Function to synchronize the index
sycn <- function(x,y){
  m <- matrix(0,length(x$Price),2)
  for(i in 1:length(x$Price)){
    for(j in 1:length(y$Price)){
      if(y$Date[j]==x$Date[i]){
        m[i,1] <- y$Price[j]
        m[i,2] <- x$Price[i]
      }
    }
  }
  return(m)
}

#Calculate the correlation with NDDUJN
NDDUE15X <- cor(as.numeric(d1$Price),as.numeric(d2$Price))
s <- sycn(d3,d1)
LMBITR <- cor(s[,1],s[,2])
NDDUNA <- cor(as.numeric(d1$Price),as.numeric(d4$Price))
NDDUPXJ <- cor(as.numeric(d1$Price),as.numeric(d5$Price))
NDDUUK <- cor(as.numeric(d1$Price),as.numeric(d6$Price))
NUUDWXUS <- cor(as.numeric(d1$Price),as.numeric(d7$Price))
s <- sycn(d3,d8)
RU10GRTR <- cor(s[,1],s[,2])
s <- sycn(d3,d9)
RU20GRTR <- cor(s[,1],s[,2])
s <- sycn(d3,d10)
S5COND <- cor(s[,1],s[,2])
s <- sycn(d3,d11)
S5ENRS <- cor(s[,1],s[,2])
