class Pipeline:

    def cleaning(self, a):
        target=a['churn'].astype(int)
        a.drop(['churn', 'phone number', 'account length'], axis=1,inplace=True)
        clist=['state', 'international plan', 'voice mail plan']
        for c in clist:
            a[c]=a[c].astype('category').cat.codes
        return a, target

    def train(self, a, b):
        self.model=gbm().fit(a, b)

    def predict(self, a):
        y_hat=self.model.predict(a)
        return y_hat

    def probs(self, a):
        probs=self.model.predict_proba(a)
        return probs

    def importance(self,a):
        self.feat_imp = pd.Series(self.model.feature_importances_).sort_values(ascending=True)

    def accuracy(self, a, b):
        self.acc=acc(a,b)
        print('The GBM model accuracy is ' + str(100*self.acc) + ' %.')
