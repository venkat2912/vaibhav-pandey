{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import warnings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# get the Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "columns_name=[\"user_id\",\"item_id\",\"rating\",\"timetamp\"]\n",
    "df=pd.read_csv(\"ml-100k/u.data\",sep=\"\\t\",names=columns_name)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>item_id</th>\n",
       "      <th>rating</th>\n",
       "      <th>timetamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>196</td>\n",
       "      <td>242</td>\n",
       "      <td>3</td>\n",
       "      <td>881250949</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>186</td>\n",
       "      <td>302</td>\n",
       "      <td>3</td>\n",
       "      <td>891717742</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>22</td>\n",
       "      <td>377</td>\n",
       "      <td>1</td>\n",
       "      <td>878887116</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>244</td>\n",
       "      <td>51</td>\n",
       "      <td>2</td>\n",
       "      <td>880606923</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>166</td>\n",
       "      <td>346</td>\n",
       "      <td>1</td>\n",
       "      <td>886397596</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   user_id  item_id  rating   timetamp\n",
       "0      196      242       3  881250949\n",
       "1      186      302       3  891717742\n",
       "2       22      377       1  878887116\n",
       "3      244       51       2  880606923\n",
       "4      166      346       1  886397596"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(99999, 4)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "943"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"user_id\"].nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1682"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df[\"item_id\"].nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "movie_title=pd.read_csv(\"ml-100k/u.item\",sep=\"\\|\",header=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1682, 24)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movie_title.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "movie_title=movie_title[[0,1]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "movie_title.columns=[\"item_id\",\"title\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>item_id</th>\n",
       "      <th>title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Toy Story (1995)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>GoldenEye (1995)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Four Rooms (1995)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Get Shorty (1995)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Copycat (1995)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>6</td>\n",
       "      <td>Shanghai Triad (Yao a yao yao dao waipo qiao) ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>7</td>\n",
       "      <td>Twelve Monkeys (1995)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>8</td>\n",
       "      <td>Babe (1995)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>9</td>\n",
       "      <td>Dead Man Walking (1995)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>10</td>\n",
       "      <td>Richard III (1995)</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   item_id                                              title\n",
       "0        1                                   Toy Story (1995)\n",
       "1        2                                   GoldenEye (1995)\n",
       "2        3                                  Four Rooms (1995)\n",
       "3        4                                  Get Shorty (1995)\n",
       "4        5                                     Copycat (1995)\n",
       "5        6  Shanghai Triad (Yao a yao yao dao waipo qiao) ...\n",
       "6        7                              Twelve Monkeys (1995)\n",
       "7        8                                        Babe (1995)\n",
       "8        9                            Dead Man Walking (1995)\n",
       "9       10                                 Richard III (1995)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movie_title.head(n=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.merge(df,movie_title,on='item_id')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>item_id</th>\n",
       "      <th>rating</th>\n",
       "      <th>timetamp</th>\n",
       "      <th>title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>99995</th>\n",
       "      <td>840</td>\n",
       "      <td>1674</td>\n",
       "      <td>4</td>\n",
       "      <td>891211682</td>\n",
       "      <td>Mamma Roma (1962)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99996</th>\n",
       "      <td>655</td>\n",
       "      <td>1640</td>\n",
       "      <td>3</td>\n",
       "      <td>888474646</td>\n",
       "      <td>Eighth Day, The (1996)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99997</th>\n",
       "      <td>655</td>\n",
       "      <td>1637</td>\n",
       "      <td>3</td>\n",
       "      <td>888984255</td>\n",
       "      <td>Girls Town (1996)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99998</th>\n",
       "      <td>655</td>\n",
       "      <td>1630</td>\n",
       "      <td>3</td>\n",
       "      <td>887428735</td>\n",
       "      <td>Silence of the Palace, The (Saimt el Qusur) (1...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99999</th>\n",
       "      <td>655</td>\n",
       "      <td>1641</td>\n",
       "      <td>3</td>\n",
       "      <td>887427810</td>\n",
       "      <td>Dadetown (1995)</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       user_id  item_id  rating   timetamp  \\\n",
       "99995      840     1674       4  891211682   \n",
       "99996      655     1640       3  888474646   \n",
       "99997      655     1637       3  888984255   \n",
       "99998      655     1630       3  887428735   \n",
       "99999      655     1641       3  887427810   \n",
       "\n",
       "                                                   title  \n",
       "99995                                  Mamma Roma (1962)  \n",
       "99996                             Eighth Day, The (1996)  \n",
       "99997                                  Girls Town (1996)  \n",
       "99998  Silence of the Palace, The (Saimt el Qusur) (1...  \n",
       "99999                                    Dadetown (1995)  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Explory Data analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pylab as plt\n",
    "import seaborn as sns\n",
    "sns.set_style('white')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "title\n",
       "Marlene Dietrich: Shadow and Light (1996)       5.0\n",
       "Prefontaine (1997)                              5.0\n",
       "Santa with Muscles (1996)                       5.0\n",
       "Star Kid (1997)                                 5.0\n",
       "Someone Else's America (1995)                   5.0\n",
       "                                               ... \n",
       "Touki Bouki (Journey of the Hyena) (1973)       1.0\n",
       "JLG/JLG - autoportrait de dÃ©cembre (1994)       1.0\n",
       "Daens (1992)                                    1.0\n",
       "Butterfly Kiss (1995)                           1.0\n",
       "Eye of Vichy, The (Oeil de Vichy, L') (1993)    1.0\n",
       "Name: rating, Length: 1664, dtype: float64"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby('title').mean()['rating'].sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "title\n",
       "Star Wars (1977)                              583\n",
       "Contact (1997)                                509\n",
       "Fargo (1996)                                  508\n",
       "Return of the Jedi (1983)                     507\n",
       "Liar Liar (1997)                              485\n",
       "                                             ... \n",
       "Man from Down Under, The (1943)                 1\n",
       "Marlene Dietrich: Shadow and Light (1996)       1\n",
       "Mat' i syn (1997)                               1\n",
       "Mille bolle blu (1993)                          1\n",
       "Ã kÃ¶ldum klaka (Cold Fever) (1994)              1\n",
       "Name: rating, Length: 1664, dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.groupby(\"title\").count()['rating'].sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "rating=pd.DataFrame(df.groupby('title').mean()['rating'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>rating</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>'Til There Was You (1997)</th>\n",
       "      <td>2.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1-900 (1994)</th>\n",
       "      <td>2.600000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>101 Dalmatians (1996)</th>\n",
       "      <td>2.908257</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12 Angry Men (1957)</th>\n",
       "      <td>4.344000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>187 (1997)</th>\n",
       "      <td>3.024390</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                             rating\n",
       "title                              \n",
       "'Til There Was You (1997)  2.333333\n",
       "1-900 (1994)               2.600000\n",
       "101 Dalmatians (1996)      2.908257\n",
       "12 Angry Men (1957)        4.344000\n",
       "187 (1997)                 3.024390"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rating.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "rating['num of rating'] = pd.DataFrame(df.groupby(\"title\").count()['rating'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>rating</th>\n",
       "      <th>num of rating</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>They Made Me a Criminal (1939)</th>\n",
       "      <td>5.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Marlene Dietrich: Shadow and Light (1996)</th>\n",
       "      <td>5.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Saint of Fort Washington, The (1993)</th>\n",
       "      <td>5.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Someone Else's America (1995)</th>\n",
       "      <td>5.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Star Kid (1997)</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Eye of Vichy, The (Oeil de Vichy, L') (1993)</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>King of New York (1990)</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Touki Bouki (Journey of the Hyena) (1973)</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Bloody Child, The (1996)</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Crude Oasis, The (1995)</th>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1664 rows Ã— 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              rating  num of rating\n",
       "title                                                              \n",
       "They Made Me a Criminal (1939)                   5.0              1\n",
       "Marlene Dietrich: Shadow and Light (1996)        5.0              1\n",
       "Saint of Fort Washington, The (1993)             5.0              2\n",
       "Someone Else's America (1995)                    5.0              1\n",
       "Star Kid (1997)                                  5.0              3\n",
       "...                                              ...            ...\n",
       "Eye of Vichy, The (Oeil de Vichy, L') (1993)     1.0              1\n",
       "King of New York (1990)                          1.0              1\n",
       "Touki Bouki (Journey of the Hyena) (1973)        1.0              1\n",
       "Bloody Child, The (1996)                         1.0              1\n",
       "Crude Oasis, The (1995)                          1.0              1\n",
       "\n",
       "[1664 rows x 2 columns]"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rating.sort_values(by=\"rating\",ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10,6))\n",
    "plt.hist(rating['num of rating'],bins=70)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.hist(rating['rating'],bins=70)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.JointGrid at 0x1d633a22eb0>"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x432 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.jointplot(x='rating',y='num of rating',data=rating,alpha=0.5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# creating movie Recommendation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>item_id</th>\n",
       "      <th>rating</th>\n",
       "      <th>timetamp</th>\n",
       "      <th>title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>196</td>\n",
       "      <td>242</td>\n",
       "      <td>3</td>\n",
       "      <td>881250949</td>\n",
       "      <td>Kolya (1996)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>63</td>\n",
       "      <td>242</td>\n",
       "      <td>3</td>\n",
       "      <td>875747190</td>\n",
       "      <td>Kolya (1996)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>226</td>\n",
       "      <td>242</td>\n",
       "      <td>5</td>\n",
       "      <td>883888671</td>\n",
       "      <td>Kolya (1996)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>154</td>\n",
       "      <td>242</td>\n",
       "      <td>3</td>\n",
       "      <td>879138235</td>\n",
       "      <td>Kolya (1996)</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>306</td>\n",
       "      <td>242</td>\n",
       "      <td>5</td>\n",
       "      <td>876503793</td>\n",
       "      <td>Kolya (1996)</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   user_id  item_id  rating   timetamp         title\n",
       "0      196      242       3  881250949  Kolya (1996)\n",
       "1       63      242       3  875747190  Kolya (1996)\n",
       "2      226      242       5  883888671  Kolya (1996)\n",
       "3      154      242       3  879138235  Kolya (1996)\n",
       "4      306      242       5  876503793  Kolya (1996)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "moviemat=df.pivot_table(index='user_id',columns='title',values='rating')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>title</th>\n",
       "      <th>'Til There Was You (1997)</th>\n",
       "      <th>1-900 (1994)</th>\n",
       "      <th>101 Dalmatians (1996)</th>\n",
       "      <th>12 Angry Men (1957)</th>\n",
       "      <th>187 (1997)</th>\n",
       "      <th>2 Days in the Valley (1996)</th>\n",
       "      <th>20,000 Leagues Under the Sea (1954)</th>\n",
       "      <th>2001: A Space Odyssey (1968)</th>\n",
       "      <th>3 Ninjas: High Noon At Mega Mountain (1998)</th>\n",
       "      <th>39 Steps, The (1935)</th>\n",
       "      <th>...</th>\n",
       "      <th>Yankee Zulu (1994)</th>\n",
       "      <th>Year of the Horse (1997)</th>\n",
       "      <th>You So Crazy (1994)</th>\n",
       "      <th>Young Frankenstein (1974)</th>\n",
       "      <th>Young Guns (1988)</th>\n",
       "      <th>Young Guns II (1990)</th>\n",
       "      <th>Young Poisoner's Handbook, The (1995)</th>\n",
       "      <th>Zeus and Roxanne (1997)</th>\n",
       "      <th>unknown</th>\n",
       "      <th>Ã kÃ¶ldum klaka (Cold Fever) (1994)</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>user_id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows Ã— 1664 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "title    'Til There Was You (1997)  1-900 (1994)  101 Dalmatians (1996)  \\\n",
       "user_id                                                                   \n",
       "1                              NaN           NaN                    2.0   \n",
       "2                              NaN           NaN                    NaN   \n",
       "3                              NaN           NaN                    NaN   \n",
       "4                              NaN           NaN                    NaN   \n",
       "5                              NaN           NaN                    2.0   \n",
       "\n",
       "title    12 Angry Men (1957)  187 (1997)  2 Days in the Valley (1996)  \\\n",
       "user_id                                                                 \n",
       "1                        5.0         NaN                          NaN   \n",
       "2                        NaN         NaN                          NaN   \n",
       "3                        NaN         2.0                          NaN   \n",
       "4                        NaN         NaN                          NaN   \n",
       "5                        NaN         NaN                          NaN   \n",
       "\n",
       "title    20,000 Leagues Under the Sea (1954)  2001: A Space Odyssey (1968)  \\\n",
       "user_id                                                                      \n",
       "1                                        3.0                           4.0   \n",
       "2                                        NaN                           NaN   \n",
       "3                                        NaN                           NaN   \n",
       "4                                        NaN                           NaN   \n",
       "5                                        NaN                           4.0   \n",
       "\n",
       "title    3 Ninjas: High Noon At Mega Mountain (1998)  39 Steps, The (1935)  \\\n",
       "user_id                                                                      \n",
       "1                                                NaN                   NaN   \n",
       "2                                                1.0                   NaN   \n",
       "3                                                NaN                   NaN   \n",
       "4                                                NaN                   NaN   \n",
       "5                                                NaN                   NaN   \n",
       "\n",
       "title    ...  Yankee Zulu (1994)  Year of the Horse (1997)  \\\n",
       "user_id  ...                                                 \n",
       "1        ...                 NaN                       NaN   \n",
       "2        ...                 NaN                       NaN   \n",
       "3        ...                 NaN                       NaN   \n",
       "4        ...                 NaN                       NaN   \n",
       "5        ...                 NaN                       NaN   \n",
       "\n",
       "title    You So Crazy (1994)  Young Frankenstein (1974)  Young Guns (1988)  \\\n",
       "user_id                                                                      \n",
       "1                        NaN                        5.0                3.0   \n",
       "2                        NaN                        NaN                NaN   \n",
       "3                        NaN                        NaN                NaN   \n",
       "4                        NaN                        NaN                NaN   \n",
       "5                        NaN                        4.0                NaN   \n",
       "\n",
       "title    Young Guns II (1990)  Young Poisoner's Handbook, The (1995)  \\\n",
       "user_id                                                                \n",
       "1                         NaN                                    NaN   \n",
       "2                         NaN                                    NaN   \n",
       "3                         NaN                                    NaN   \n",
       "4                         NaN                                    NaN   \n",
       "5                         NaN                                    NaN   \n",
       "\n",
       "title    Zeus and Roxanne (1997)  unknown  Ã kÃ¶ldum klaka (Cold Fever) (1994)  \n",
       "user_id                                                                        \n",
       "1                            NaN      4.0                                 NaN  \n",
       "2                            NaN      NaN                                 NaN  \n",
       "3                            NaN      NaN                                 NaN  \n",
       "4                            NaN      NaN                                 NaN  \n",
       "5                            NaN      4.0                                 NaN  \n",
       "\n",
       "[5 rows x 1664 columns]"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "moviemat.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>rating</th>\n",
       "      <th>num of rating</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Star Wars (1977)</th>\n",
       "      <td>4.358491</td>\n",
       "      <td>583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Contact (1997)</th>\n",
       "      <td>3.803536</td>\n",
       "      <td>509</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fargo (1996)</th>\n",
       "      <td>4.155512</td>\n",
       "      <td>508</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Return of the Jedi (1983)</th>\n",
       "      <td>4.007890</td>\n",
       "      <td>507</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Liar Liar (1997)</th>\n",
       "      <td>3.156701</td>\n",
       "      <td>485</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                             rating  num of rating\n",
       "title                                             \n",
       "Star Wars (1977)           4.358491            583\n",
       "Contact (1997)             3.803536            509\n",
       "Fargo (1996)               4.155512            508\n",
       "Return of the Jedi (1983)  4.007890            507\n",
       "Liar Liar (1997)           3.156701            485"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rating.sort_values('num of rating',ascending=False).head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "user_id\n",
       "1    5.0\n",
       "2    5.0\n",
       "3    NaN\n",
       "4    5.0\n",
       "5    4.0\n",
       "Name: Star Wars (1977), dtype: float64"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "starwar_user_rating=moviemat['Star Wars (1977)']\n",
    "starwar_user_rating.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "similar_to_starwars=moviemat.corrwith(starwar_user_rating)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "title\n",
       "'Til There Was You (1997)                0.872872\n",
       "1-900 (1994)                            -0.645497\n",
       "101 Dalmatians (1996)                    0.211132\n",
       "12 Angry Men (1957)                      0.184289\n",
       "187 (1997)                               0.027398\n",
       "                                           ...   \n",
       "Young Guns II (1990)                     0.228615\n",
       "Young Poisoner's Handbook, The (1995)   -0.007374\n",
       "Zeus and Roxanne (1997)                  0.818182\n",
       "unknown                                  0.723123\n",
       "Ã kÃ¶ldum klaka (Cold Fever) (1994)            NaN\n",
       "Length: 1664, dtype: float64"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "similar_to_starwars"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])\n",
    "corr_starwars.dropna(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>correlation</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>'Til There Was You (1997)</th>\n",
       "      <td>0.872872</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1-900 (1994)</th>\n",
       "      <td>-0.645497</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>101 Dalmatians (1996)</th>\n",
       "      <td>0.211132</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12 Angry Men (1957)</th>\n",
       "      <td>0.184289</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>187 (1997)</th>\n",
       "      <td>0.027398</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                           correlation\n",
       "title                                 \n",
       "'Til There Was You (1997)     0.872872\n",
       "1-900 (1994)                 -0.645497\n",
       "101 Dalmatians (1996)         0.211132\n",
       "12 Angry Men (1957)           0.184289\n",
       "187 (1997)                    0.027398"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corr_starwars.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>correlation</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Hollow Reed (1996)</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Commandments (1997)</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cosi (1996)</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>No Escape (1994)</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Stripes (1981)</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Star Wars (1977)</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Man of the Year (1995)</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Beans of Egypt, Maine, The (1994)</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Old Lady Who Walked in the Sea, The (Vieille qui marchait dans la mer, La) (1991)</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Outlaw, The (1943)</th>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                    correlation\n",
       "title                                                          \n",
       "Hollow Reed (1996)                                          1.0\n",
       "Commandments (1997)                                         1.0\n",
       "Cosi (1996)                                                 1.0\n",
       "No Escape (1994)                                            1.0\n",
       "Stripes (1981)                                              1.0\n",
       "Star Wars (1977)                                            1.0\n",
       "Man of the Year (1995)                                      1.0\n",
       "Beans of Egypt, Maine, The (1994)                           1.0\n",
       "Old Lady Who Walked in the Sea, The (Vieille qu...          1.0\n",
       "Outlaw, The (1943)                                          1.0"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corr_starwars.sort_values('correlation',ascending=False).head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>rating</th>\n",
       "      <th>num of rating</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>'Til There Was You (1997)</th>\n",
       "      <td>2.333333</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1-900 (1994)</th>\n",
       "      <td>2.600000</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>101 Dalmatians (1996)</th>\n",
       "      <td>2.908257</td>\n",
       "      <td>109</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12 Angry Men (1957)</th>\n",
       "      <td>4.344000</td>\n",
       "      <td>125</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>187 (1997)</th>\n",
       "      <td>3.024390</td>\n",
       "      <td>41</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Young Guns II (1990)</th>\n",
       "      <td>2.772727</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Young Poisoner's Handbook, The (1995)</th>\n",
       "      <td>3.341463</td>\n",
       "      <td>41</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Zeus and Roxanne (1997)</th>\n",
       "      <td>2.166667</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unknown</th>\n",
       "      <td>3.444444</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ã kÃ¶ldum klaka (Cold Fever) (1994)</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1664 rows Ã— 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                         rating  num of rating\n",
       "title                                                         \n",
       "'Til There Was You (1997)              2.333333              9\n",
       "1-900 (1994)                           2.600000              5\n",
       "101 Dalmatians (1996)                  2.908257            109\n",
       "12 Angry Men (1957)                    4.344000            125\n",
       "187 (1997)                             3.024390             41\n",
       "...                                         ...            ...\n",
       "Young Guns II (1990)                   2.772727             44\n",
       "Young Poisoner's Handbook, The (1995)  3.341463             41\n",
       "Zeus and Roxanne (1997)                2.166667              6\n",
       "unknown                                3.444444              9\n",
       "Ã kÃ¶ldum klaka (Cold Fever) (1994)     3.000000              1\n",
       "\n",
       "[1664 rows x 2 columns]"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rating"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>correlation</th>\n",
       "      <th>num of rating</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>'Til There Was You (1997)</th>\n",
       "      <td>0.872872</td>\n",
       "      <td>9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1-900 (1994)</th>\n",
       "      <td>-0.645497</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>101 Dalmatians (1996)</th>\n",
       "      <td>0.211132</td>\n",
       "      <td>109</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12 Angry Men (1957)</th>\n",
       "      <td>0.184289</td>\n",
       "      <td>125</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>187 (1997)</th>\n",
       "      <td>0.027398</td>\n",
       "      <td>41</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                           correlation  num of rating\n",
       "title                                                \n",
       "'Til There Was You (1997)     0.872872              9\n",
       "1-900 (1994)                 -0.645497              5\n",
       "101 Dalmatians (1996)         0.211132            109\n",
       "12 Angry Men (1957)           0.184289            125\n",
       "187 (1997)                    0.027398             41"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corr_starwars=corr_starwars.join(rating['num of rating'])\n",
    "corr_starwars.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>correlation</th>\n",
       "      <th>num of rating</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Star Wars (1977)</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>583</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Empire Strikes Back, The (1980)</th>\n",
       "      <td>0.747981</td>\n",
       "      <td>367</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Return of the Jedi (1983)</th>\n",
       "      <td>0.672556</td>\n",
       "      <td>507</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Raiders of the Lost Ark (1981)</th>\n",
       "      <td>0.536117</td>\n",
       "      <td>420</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Austin Powers: International Man of Mystery (1997)</th>\n",
       "      <td>0.377433</td>\n",
       "      <td>130</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Edge, The (1997)</th>\n",
       "      <td>-0.127167</td>\n",
       "      <td>113</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>As Good As It Gets (1997)</th>\n",
       "      <td>-0.130466</td>\n",
       "      <td>112</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Crash (1996)</th>\n",
       "      <td>-0.148507</td>\n",
       "      <td>128</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>G.I. Jane (1997)</th>\n",
       "      <td>-0.176734</td>\n",
       "      <td>175</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>First Wives Club, The (1996)</th>\n",
       "      <td>-0.194496</td>\n",
       "      <td>160</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>334 rows Ã— 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                    correlation  num of rating\n",
       "title                                                                         \n",
       "Star Wars (1977)                                       1.000000            583\n",
       "Empire Strikes Back, The (1980)                        0.747981            367\n",
       "Return of the Jedi (1983)                              0.672556            507\n",
       "Raiders of the Lost Ark (1981)                         0.536117            420\n",
       "Austin Powers: International Man of Mystery (1997)     0.377433            130\n",
       "...                                                         ...            ...\n",
       "Edge, The (1997)                                      -0.127167            113\n",
       "As Good As It Gets (1997)                             -0.130466            112\n",
       "Crash (1996)                                          -0.148507            128\n",
       "G.I. Jane (1997)                                      -0.176734            175\n",
       "First Wives Club, The (1996)                          -0.194496            160\n",
       "\n",
       "[334 rows x 2 columns]"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corr_starwars[corr_starwars['num of rating']>100].sort_values('correlation',ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## predict Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_movies(movie_name):\n",
    "    movie_user_rating=moviemat[movie_name]\n",
    "    similar_to_movie=moviemat.corrwith(movie_user_rating)\n",
    "    \n",
    "    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])\n",
    "    corr_movie.dropna(inplace=True)\n",
    "    \n",
    "    #join first\n",
    "    corr_movie=corr_movie.join(rating['num of rating'])\n",
    "    pred=corr_movie[corr_movie['num of rating']>100].sort_values('correlation',ascending=False)\n",
    "    \n",
    "    return pred\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred=predict_movies(\"Titanic (1997)\")[1:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>correlation</th>\n",
       "      <th>num of rating</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>River Wild, The (1994)</th>\n",
       "      <td>0.497600</td>\n",
       "      <td>146</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Abyss, The (1989)</th>\n",
       "      <td>0.472103</td>\n",
       "      <td>151</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Bram Stoker's Dracula (1992)</th>\n",
       "      <td>0.443560</td>\n",
       "      <td>120</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>True Lies (1994)</th>\n",
       "      <td>0.435104</td>\n",
       "      <td>208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>William Shakespeare's Romeo and Juliet (1996)</th>\n",
       "      <td>0.430243</td>\n",
       "      <td>106</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               correlation  num of rating\n",
       "title                                                                    \n",
       "River Wild, The (1994)                            0.497600            146\n",
       "Abyss, The (1989)                                 0.472103            151\n",
       "Bram Stoker's Dracula (1992)                      0.443560            120\n",
       "True Lies (1994)                                  0.435104            208\n",
       "William Shakespeare's Romeo and Juliet (1996)     0.430243            106"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
