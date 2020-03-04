
import os

runs = [ 'Run-Alcohol_consumption','Run-Carbonated_beverages','Run-Chocolate','Run-Coffee','Run-Diagnosis',
         'Run-Gender','Run-Occupation_status','Run-Smoker','Run-Soft_cheese','Run-Tomatoes' ]
runs = [ 'Run-Diagnosis_Simple', 'Run-Citrus_fruits' ]
measures = ['accuracy','precision','recall','f1','roc_auc','cap_auc','time']
measures = ['accuracy','precision','recall','f1','roc_auc' ]

with open('Results-new.md','w') as file:
    for run in runs:
        print('',file=file)
        print('### Results for ' + run[4:],file=file)
        print('',file=file)
        for measure in measures:
            f = '../../results/' + run + '/comparisons/' + measure + '.cv.png'
            n = ('Result-' + run + '-' + measure).replace(' ','_')
            s = '![' + n + '](' + f + ')'

            if not os.path.isfile(f):
                print("FILE '" + f + "' is missing.")
            print(s,file=file)
        print('',file=file)
        print('',file=file)

            #![Result](../../results/Run-001/comparisons/accuracy.cv.png)