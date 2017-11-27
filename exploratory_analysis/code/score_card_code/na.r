yqjr_outlier<-function(data,Variable_name,outlier_limit=0.05){
    IQR_df<-as.data.frame(list(Q25=NA,Q75=NA,IQR=NA,IQR1.5=NA,outlier_num=NA,outlier_percent=NA,IQR_Bound=NA,IQR_Bound_Value=NA,Q97=NA,Q98=NA,Q99=NA,Exceed=NA,Advice=NA))
    IQR_df<-IQR_df[-1,]
    for (i in Variable_name){
        Q<-quantile(data[[i]],c(0.25,0.75),na.rm = T)
        Q25<-Q[[1]]
        Q75<-Q[[2]]
        IQR<-Q[[2]]-Q[[1]]
        IQR1.5<-1.5*IQR
        IQR_Bound<-Q[[2]]+IQR1.5
        IQR_Bound_Value<-max(data[data[,i]<=IQR_Bound,i])
        outlier_num<-nrow(data[data[[i]]>IQR_Bound_Value & !is.na(data[[i]]),])
        outlier_percent<-outlier_num/nrow(data)
        Q97<-quantile(data[[i]],0.97,na.rm = T);
        Q98<-quantile(data[[i]],0.98,na.rm = T);
        Q99<-quantile(data[[i]],0.99,na.rm = T);
        Exceed<-outlier_percent>outlier_limit
        Advice<-ifelse(Exceed,"Q97-Q99",IQR_Bound_Value)
        temp_df<-as.data.frame(list(Q25=Q25,Q75=Q75,IQR=IQR,IQR1.5=IQR1.5,outlier_num=outlier_num,outlier_percent=outlier_percent,IQR_Bound=IQR_Bound,IQR_Bound_Value=IQR_Bound_Value,Q97=Q97,Q98=Q98,Q99=Q99,Exceed=Exceed,Advice=Advice))
        row.names(temp_df)<-i
        IQR_df<-rbind(IQR_df,temp_df)
    }
  return(IQR_df)
}


yqjr_outlier_fix_AsChoice<-function(data,Variable_name,choice_series,outlier_limit=0.05){
    IQR_df<-yqjr_outlier(data,Variable_name,outlier_limit)
    exceed<-row.names(IQR_df[IQR_df$Exceed==TRUE,])
    for (i in exceed){
          data[data[[i]]>as.numeric(IQR_df[i,"IQR_Bound"]),i]<-as.numeric(IQR_df[i,choice_series[which(i %in% exceed)]])
    }
    return(data)
}


data = read.csv(file.choose(), sep=',')
IQR_df = yqjr_outlier(data, names(data)[-1:-4])
print(IQR_df[IQR_df$Exceed==TRUE,])
choice_series = c('Q97','Q97','Q97','Q97','Q97','Q97','Q97','Q97','Q97','Q97')
result = yqjr_outlier_fix_AsChoice(data, names(data)[-1:-4], choice_series)
#print(result[1:5,])
write.csv(result, ".\\cleaned_detail_info_R.csv", row.names=F)

#IQR_df1 = yqjr_outlier(result, names(result)[-1:-4])
#print(IQR_df1[IQR_df1$Exceed==TRUE,])


