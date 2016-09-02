require(M4comp);
nseries <- length(M4);

df = data.frame(K = character(),
		ID = character(),
		TYPE = character(),
		H = character(),
		PERIOD = character(),
		UNITS = character(),
		N = character(),
		PAST = character(),
		FUTURE = character());

class.data  <- sapply(df, class)
factor.vars <- class.data[class.data == "factor"]
for (colname in names(factor.vars))
{
    df[,colname] <- as.character(df[,colname])
}

for(i in 1:nseries) {
  xx = M4[[i]];
  past1 = paste(M4[[i]]$past , collapse = ",")
  future1 = paste(M4[[i]]$future, collapse = ",")
  line1 = c("SERIES", M4[[i]]$id, M4[[i]]$type, M4[[i]]$H, M4[[i]]$period, M4[[i]]$units, M4[[i]]$n, past1 , future1, stringsAsFactors=FALSE);
  df[i,] = line1;
  #print(line1);
}

#print(df);
write.csv(df, file = "M4Comp.csv", row.names=FALSE, na="")
