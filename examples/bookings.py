import sys
sys.path.insert(0, '../bayestest')

import bayestest as bt


Test = bt.bayesTest()


Test.prior(alpha=2, beta=25)
Test.prior(alpha=2, beta=15, revenue=True)

Test.add(visitors=1000, successes=10, revenue=100.00, control=True, name='Control')
Test.add(visitors=1000, successes=15, revenue=158.00, control=False, name='V1')
Test.add(visitors=1000, successes=20, revenue=198.00, control=False, name='V2')
Test.infer(samples=10000)
results = Test.samples()
print(results[1])
Test.summary()
