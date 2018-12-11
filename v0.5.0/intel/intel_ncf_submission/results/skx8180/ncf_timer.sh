
# This script calculates ncf timing correctly, which is 1790.79.

grep "true" ncf/* | sort -V | awk 'BEGIN {FS="::::"} {print $1} NR==50 {exit}' \
    > first_50_success_file

while read F;
do
  grep "Finish training in" $F
done < first_50_success_file | sort -V | \
    awk 'BEGIN {sum = 0} {if (NR != 1 && NR != 50) {sum += $4}} \
    END {print sum / (NR - 2)}'
