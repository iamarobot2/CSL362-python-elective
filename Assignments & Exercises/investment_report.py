invested_amount = float(input('Enter Investment Amount:'))
years = int(input('Enter the number of years:'))
interest_rate = int(input('Enter rate as a %:'))/100
print('Years','Starting Balance','Interest','Ending balance',sep='|',end='|\n')
total_interest = 0
for i in range(years):
    interest = invested_amount * interest_rate
    total_interest+=interest
    ending_balance = invested_amount + interest
    print('%5i'%i,'%16.2f'%invested_amount,'%8.2f'%interest,'%14.2f'%ending_balance,sep='|',end='|\n')
    invested_amount = ending_balance
print('Ending balance:Rs.'+'%.2f'%invested_amount)
print('Total interest earned:Rs.'+'%.2f'%total_interest)