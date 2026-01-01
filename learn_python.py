# %%
import numpy as np
import pandas as pd

# %%
import pandas as pd
a = np.array([1,2,3,4,5]) # create 1D array
# %%
b = np.array ([[1,2,3], [1,3,6]])
b
# %%
ze = np.zeros((2,3))
ze
# %%
ones = np.ones((2,3))
ones
# %%
rand = np.random.rand (2,3)
rand
# %%
aa = np.arange (0,10,1)
aa
# %%
ln = np.linspace(0,1,10)
ln
# %%
x = np.array([[1.2, 3.4, 5.6],
              [7.8, 9.0, 1.1]])

print(x.shape)   # shape of array
print(x.ndim)    # number of dimensions
print(x.dtype)   # data type

# %%
a = np.array ([1,2,3,4,5,6,7,8,9])
print(a[0])
# %%
print(a[1:3])
print(a[:2])
print(a[1:])
print(a[::3])
# %%
b = np.array([10,20,30,40,50,60])
mask = b>20
print (mask)
print(b[mask])
# %%
print(b[(b>10) & (b<30)])
# %%
b[b<40] = 2
print(b)
# %%
c = np.array([1,2,3,4,5,6,7])
inx = [0,2,4]
print(c[inx])
# %%
v = np.array([1,np.pi, np.pi/2,20,100])
print(np.sum(v))
print(np.log(v))
# %%
n = np.array ([1,2,3,4,5,8])
c = n.reshape(3,-1)
print(c)
# %%
m = np.array([[1, 2, 3],
              [4, 5, 6]])

r = m.ravel()
f = m.flatten()
print(r)
print(f)
# %%
c = m[1:3].copy()
print(c)
c[0] = 555

# %%
x = np.array([1, 2, 3, 4])
y = x[1:]

print(np.shares_memory(x, y))

# %%
q = np.random.rand(5)
print(q)
q1 = np.random.uniform(10,20,5)
print(q1)
# %%
q2 = np.random.choice ([1,2,3,4,5,6] , 6)
print(q2)
# %%
np.random.choice([0, 1], size=10, p=[0.7, 0.3])

# %%
N = 100_000
samples = np.random.normal(0, 1, N)

prob = np.mean(samples > 1.96)
print(prob)

# %%
steps = np.random.choice([-1, 1], size=1000)
walk = steps.cumsum()
print(walk)
# %%
n = 10000
e = np.random.normal(0,1,n)
v = np.mean(e>.70)
print(v)
# %%
A = np.array([[4, 7],
              [2, 6]])

A_inv = np.linalg.inv(A)

print(A @ A_inv)

# %%
b = np.array([1, 0])
x = np.linalg.solve(A, b)
print (x)

# %%
M = np.array([[2, 0],
              [0, 3]])

vals, vecs = np.linalg.eig(M)
print(vals, vecs)
# %%
X = np.array([[1, 1],
              [1, 2],
              [1, 3]])

y = np.array([1, 2, 2.5])

beta = np.linalg.inv(X.T @ X) @ X.T @ y
print(beta)
# %%


####################### start pandas ###################
#########################################################
import pandas as pd
p = pd.Series([1,2,3,4,5,6])
p

# %%
p1 = pd.Series([1,2,3,4], index= ['m','n','mm', 'qw'])
p1
# %%
p1['m']
p1[0]
# %%
p1/10
# %%
df = pd.DataFrame({
    "pro":['mn', "mm", "mb"], 
    "in": [1,2,3],
    "index": ["h","h1","h2"]
})
df
# %%
df.describe()
# %%
df.info()
# %%
df[["pro","in"]]
# %%
df.iloc[0:2]
# %%
df[df["in"] == 1]
# %%
df[df["pro"] =="mm"]
# %%
 df = pd.DataFrame({
  "ostan":["teh", "mar" , "ir" ,"jo"],
  "income" :[12,20,30,40],
  "edu" :[1,2,3,4]
 })
df
# %%
df1 = df.set_index("ostan")
df1
# %%
df1.loc["teh"]
# %%
df1.iloc["teh"]
# %%
df1.iloc[0]
# %%
df1.index.name = "pro"
df1
# %%
df1[
 (df1["pro"] == "teh")&
 (df1["income"] ==10)
]
# %%
df = pd.DataFrame({
    "province": ["Tehran", "Tehran", "Isfahan", "Isfahan", "Mashhad"],
    "income": [120, 95, 80, 85, 70],
    "education_years": [16, 14, 12, 13, 11],
    "gender": ["M", "F", "F", "M", "F"]
})
df

# %%
df.groupby("province") ["income"].mean()
# %%
df.groupby("province") ["income"].agg(["mean","count"])

# %%
df.groupby(["province", "gender"])["income"].mean().unstack()

# %%
g = df.groupby("province")["income"].mean().reset_index()
g

# %%
df["income_centered"] = (
    df["income"] -
    df.groupby("province")["income"].transform("mean")
)
df

# %%
people = pd.DataFrame({
    "person_id": [1, 2, 3, 4],
    "province": ["Tehran", "Isfahan", "Tehran", "Mashhad"],
    "education_years": [16, 12, 14, 11]
})
people

# %%
income = pd.DataFrame({
    "person_id": [1, 2, 4],
    "income": [120, 80, 70]
})
income

# %%
pd.merge(people,income, on = "person_id", how = "inner")
# %%
import pandas as pd
# %%
people_idx = people.set_index("person_id")
income_idx = income.set_index("person_id")

people_idx.join(income_idx)

# %%
import numpy as np
df = pd.DataFrame({
    "province": ["Tehran", "Isfahan", "Tehran", "Mashhad"],
    "income": [120, np.nan, 95, np.nan],
    "education_years": [16, 12, np.nan, 11]
})
df


# %%
df.isna()


# %%
df.isnull()

# %%
df.isna().sum()

# %%
df.isna().any()

# %%
df.dropna()

# %%
df["income"].fillna(0)

# %%
df.dropna(how="all")

df.dropna(subset=["income"])


# %%
df.isna()
# %%
df.fillna(0)
# %%
import numpy as np

# %%
import pandas as pd
# %%
df.isna()
# %%
df.fillna(0)
# %%
df.dropna()
# %%
df.dropna(how='all')
# %%
df.dropna(subset = ['income'])
# %%
df['income'].fillna(df['income'].mean())
# %%
df.isna().sum()

# %%
df = pd.DataFrame({
    "province": ["Tehran", "Tehran", "Isfahan", "Isfahan", "Mashhad"],
    "gender": ["M", "F", "M", "F", "F"],
    "income": [120, 95, 80, 85, 70]
})
df

# %%
pd.pivot_table(df, values= 'income', index = 'province', columns = 'gender', aggfunc = "mean")
# %%
df.pivot(
    index="province",
    columns="gender",
    values="income"
)

# %%
g= df.groupby(['province' , 'gender'])['income'].mean()
g
# %%
g.unstack()
# %%
g.unstack().stack()
# %%
wide = pd.DataFrame({
    "province": ["Tehran", "Isfahan"],
    "M": [120, 80],
    "F": [95, 85]
})
wide

# %%
pd.melt(
    wide,
    id_vars="province",
    value_vars=["M", "F"],
    var_name="gender",
    value_name="income"
)

# %%
table = (
    df
    .groupby(["province", "gender"])["income"]
    .mean()
    .reset_index()
    .pivot(index="province", columns="gender", values="income")
)
table

# %%
