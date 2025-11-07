library(readr)
library(dplyr)
library(feather)

directory <- '../../data'
srv_out = c(
  "NHANES_1999_2000", 
  "NHANES_2001_2002", 
  "NHANES_2003_2004", 
  "NHANES_2005_2006", 
  "NHANES_2007_2008", 
  "NHANES_2009_2010", 
  "NHANES_2011_2012", 
  "NHANES_2013_2014", 
  "NHANES_2015_2016", 
  "NHANES_2017_2018", 
  "NHANES_III"
)

read_nhanes_dat <- function (infile) {
  # read in the fixed-width format ASCII file
  file_base <- basename(infile)
  survey    <- str_replace(file_base, "_MORT_2019_PUBLIC\\.dat$", "")
  year      <- str_extract(file_base, "(?<=NHANES_)([0-9]{4}_[0-9]{4}|III)")
  dsn <- read_fwf(file=infile,
                  col_types = "iiiiiiii",
                  fwf_cols(SEQN = c(1,6),
                           eligstat = c(15,15),
                           mortstat = c(16,16),
                           ucod_leading = c(17,19),
                           diabetes = c(20,20),
                           hyperten = c(21,21),
                           permth_int = c(43,45),
                           permth_exm = c(46,48)
                  ),
                  na = c("", ".")
  ) %>% 
    mutate (file_name = file_base,
            survey = survey,
            year = year)
}

dats <- vector(mode="list", length=length(srv_out))
for(survey_dat in seq_along(srv_out)) {
  filenme <- sprintf("%s_MORT_2019_PUBLIC.dat", srv_out[survey_dat])
  filein <- file.path(directory, filenme)
  dats[[survey_dat]] <- read_nhanes_dat(filein)
}
mortality_data <- dats %>% bind_rows() #135310
saveRDS(mortality_data, file="../../data/mortality_data_III-1999-2018_2019.rds")
