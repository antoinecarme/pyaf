TRS="transf_ transf_cumsum transf_diff transf_exp transf_inv     transf_log     transf_pow3    transf_sqr     transf_sqrt"
TRS="transf_inv transf_log transf_pow3 transf_sqr transf_sqrt"

for tr in $TRS 
do
 make -f tests/artificial/Makefile -j 12 $tr  > tests/artificial/log.$tr
done

