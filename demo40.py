# calculating probability

"""
keywords in emails
           spam    non-spam
free       0.7     0.2
now        0.6     0.3
low        0.2     0.5
economics  0.1     0.6
"""
spam = .6 * .7 * .1 * .7
non_spam = .3 * .2 * .6 * .3
print(spam, non_spam)

"""
COVID-19 test
sick people: 95% positive test result
healthy people: 6% positive test result
sick rate: 2%
"""

sick = 0.95 * 0.02
healthy = 0.06 * 0.98
# k*(sick+health)=1 ==> k=1/(sick+health)
k = 1 / (sick + healthy)
print('sick', sick)
print('healthy', healthy)
print('k', k)
print("answer", k * sick)  # positive and sick
