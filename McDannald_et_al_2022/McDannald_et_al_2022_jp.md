# On-the-fly autonomous control of neutron diffraction via physics-informed Bayesian active learning

## ABSTRACT

我々は、自律型中性子回折実験装置ANDiEを開発・配備し、中性子回折実験をライブで自律的に制御することを初めて実証した。
中性子散乱は、物質の磁気構造と挙動を調べるためのユニークで汎用性の高い特性評価技術である。
しかし、世界の中性子散乱実験施設は限られており、また、そのような施設では常に過剰な需要が発生している。
我々は、機械学習による測定パラメータ空間の自律的なナビゲーションを実装することで、中性子回折実験に必要な時間を大幅に短縮することを実証する。
事前の科学的知識とベイズ型能動学習によって、測定の順序を動的に制御する。
ANDiEは、MnOとFe1.09Teの磁気秩序転移を実験的に決定し、同時に測定効率を5倍向上させることができることを示した。
さらに、仮説検証の後処理として、ANDiEは可能性のある物理モデルの集合から転移挙動を決定することができた。
ANDiEの能動的学習アプローチは、様々な中性子実験に広く適用でき、材料探索を加速するツールとして中性子散乱の扉を開くことができる。

## I. INTRODUCTION

ベイズ型自律的物理科学は、急速に発展している新しい分野であり、知識の獲得プロセスを大幅に加速することで、科学的手法全体に革命をもたらす可能性を秘めている。
自律的物理科学では、機械学習が実験システムを制御し、最も情報量の多い実験を選択し、実行することを繰り返している。
実験室における自律的実験の最初の具体例としては、移動型ロボット化学者1、化学合成のための自動運転実験室2、自律的機械設計システム3、カーボンナノチューブ成長のための最適化ツール4などがある。
最近では、自律型放射光回折のデモンストレーション5,6が行われ、人工知能（AI）主導の実験の舞台は、割り当てられた実験ビームタイムが貴重な商品である国立研究所のビームライン環境へと広がっている。

特に中性子散乱では、測定時間の短縮とリソースの削減が求められている。
中性子回折は、磁気秩序パラメータを直接観測できる数少ない測定技術の一つである。
磁気秩序パラメータは、磁気秩序状態から無秩序状態への遷移挙動を記述する。
この磁気秩序転移がどのように、どのような温度で起こるかを理解することは、新しい磁性材料の発見にとって非常に重要である。
そのため、中性子科学施設のビームタイムは非常に重要視されている。
しかし、この強力な技術は、世界でもほんの一握りの施設でしか利用できず、需要が供給をはるかに上回っているのが現状である。

オーダーパラメータを研究するビームライン実験では、一般的に、ユーザーが適切な情報を見逃さないようにと、広いサンプリング範囲と高いサンプリング周波数を定義し、アドホックに定義された網羅的な測定を行う。
現在、これらの測定のスケジューリングについて、普遍的に合意された方法はない。
このアドホックなスケジューリングは、回折信号が強く、試料がよく知られている場合にはあまり問題にはならないが、信号が小さい試料（小さな磁気モーメント、小さな結晶、薄い膜など）や試料についてほとんど知られていない場合には効果がなく、無駄が生じる可能性がある。
この研究の目的の一つは、測定値から引き出せる統計的推論に基づいて測定値を選択することをアルゴリズム的に定式化することである。
従来のアドホックスケジュールでは、全データが収集されると、専門家が計算ツールを用いてデータを解析し、ターゲットパラメータを抽出する。
高フラックスの散乱実験（例えば、放射光X線回折測定）では、収集されたデータの大量かつ高速なため、適切な科学的情報を迅速に解明するための機械学習が必要となる7,8。
しかし、中性子回折実験のフラックスは一般的にはるかに低く、したがって、大規模なデータセットを収集することは面倒で時間がかかるものであった。
この課題は、典型的な網羅的手法では必要な量をはるかに超える測定値が得られるオーダーパラメーター調査に代表される。
このような課題に対して、機械学習は、自律制御によって後続の各実験をその場で最適化することで、知識獲得を加速する役割を果たすことができる。
本研究では、自律型中性子回折実験装置（ANDiE）を開発し、磁気転移の挙動を自己主導的に探索し、測定効率を大幅に向上させることに成功した。

この研究では、最適な実験設計（適応設計）に特化したAIの一分野である能動学習（AL）9と確率的モデリング／プログラミングを組み合わせることで、測定選択をその場でガイドすることができる。
最近の研究では、インシリコの自律的なキャンペーンにより、ベイズ手法を使用して中性子散乱測定の順序を決定できることが示されている10。
この研究では、事前に収集したデータ間を補間するためにガウス過程を使用することの有効性を示している。
本研究では、中性子散乱と磁気物理の知識を取り入れた物理ベースのアプローチによる自律的なライブ中性子散乱実験を初めて実証した。
物理ベースの知識をマルコフ連鎖モンテカルロ法（MCMC）のフレームワークに組み込むことで、ANDiEはその場で補間・外挿を行うことができるようになった。
この手法により、測定データの収集からモデルの予測に至るまで、プロセス全体を通じて不確実性を把握し、伝播させることができる。
さらに、ANDiEのMCMCフレームワークには先行物理が組み込まれており、解析結果は物理的に実現可能なものに限定されます。
これは、科学的AI（SciAI）という新分野の設計指針に沿ったものである11。
このようなSciAIによる自律的な実験は、中性子回折実験に必要な時間を大幅に短縮し、装置や専門家の時間を軽減することで、ある施設での実験を増やしたり、他の方法では実現不可能な実験を可能にする可能性を持っています。
私たちは、ANDiEが自律的にライブ中性子回折実験を行い、ネール温度（TN）を発見し、その後の試料の磁気構造の温度依存性に関する仮説検証を、通常必要な測定数の約5分の1で行うことに成功したことを示している。

アクティブラーニングのルーツは18世紀にまで遡り、ラプラスが天体の研究にALを用いたことにある12。
ベイズ最適化などのベイズAL手法は、予測値と関連する不確実性の両方を出力するために、確率的なモデリングを組み込んでいる。
これらの方法は、研究室やインシリコで未知の機能を最適化する科学者を導くのに特に有用である13-19。
材料科学の分野では、材料探索を加速させるために、自律的な実験が急務となっている20。
自律システムの原動力として、ALは、従来のAI最適化スキームを用いて、材料加工条件の最適化4,21,22、試料の特性評価6、技術的応用のためのポリマーや有機分子の組成1,2,23に有効であることが示されている。
自律システムにおけるこれらの取り組みの多くは、主に化学に焦点を当てたものであった24-26。
固体材料の自律システムという新分野のレビューでは、材料探索を加速し、複雑な材料と特性の関係を解明する上で、大きな期待が寄せられていることが示されている27,28。
これらのAIツールに事前の物理的知識をエンコードすることで、SciAIは解釈可能性を維持しつつ、知識の取得をさらに加速させるという大きな利点を有している。
例えば、材料探索と最適化のための閉ループ自律システム（CAMEO）アルゴリズムは、位相マッピングとX線回折の知識をエンコードし、材料最適化の加速において非SciAI手法よりも優れていることが実証され、クラス最高の相変化メモリ材料を発見する結果となった5。

ANDiEは、ベイズ型SciAIを採用し、中性子回折測定から磁気秩序変数の決定を高速化した。
磁気秩序変数は、磁気秩序転移の振る舞いを記述するものである。
例えば、1次転移を示す反強磁性体は温度で秩序変数が不連続になり、2次転移を示す反強磁性体は温度で秩序変数が連続になり、1次微分が温度で不連続になるといった現象が見られる。
可能であれば、事前の磁気測定、熱容量測定、輸送測定から、磁気秩序転移が示唆されることがある。
一方、磁気秩序は中性子回折で直接観測することができる。秩序状態における磁気モーメントから散乱する中性子によって、ブラッグ回折強度が増加するからである。
このため中性子回折は、前述のバルク評価法では測定が困難な物質（薄膜試料や不純物相を含む試料など）の磁気秩序化を研究することができる。
したがって、中性子回折は磁気秩序パラメータとTNを決定するための最も決定的な測定技術である。
ANDiEは、TNの事前分布を知るためにあらゆる予備的な特性評価測定を利用することができ、それにより、情報量の多いデータと正しいTN値をより早く収束させることが可能になる。
さらに、この中性子回折のプロセスにはいくつかのよく知られた物理原理があり、これらは秩序変数の自律的発見のためのアルゴリズムに組み込まれています。
これらには、回折強度がポアソン型の不確かさを持つこと、回折ピークが疑似ボイグトプロファイルでよく記述されること、回折強度の磁気成分が磁化の二乗に関係すること、などが含まれる29。
重要なのは、磁化の温度依存性が数種類のモデルに従うといえることである。
事前の物理的知識がどのように符号化されるかについての説明は、第II部にある。

回折実験は、まず転移温度以上から5Kのベース温度まで冷却することから始められる。
その後の各反復において、物理ベースの不確定性定量化により、磁気構造ピークパラメータとその温度依存性の知識を最大化するために、その後の等温回折測定の選択をガイドする。
この手法は、磁気構造解析のための等温ベイズ推論ステップと、温度依存性解析のための第2の熱ベイズ推論ステップを組み合わせている。
従来の能動学習スキームでは、最適な測定を行うために任意の温度を選択することが可能であったが、この課題ではそのようなスキームは不可能である。
一次相転移はヒステレティックであるため、測定されるTNは試料が温まっているか冷えているかに依存する。
さらに、TN以下の秩序変数はこのヒステレティックな性質のため、一度試料をベース温度以上に温めたら、秩序変数が同じ状態になるように冷却する前にTNより十分に高く温めなければならない。
この手順は法外にコストがかかる。特に、実験の目的がTNを発見することであることを考えると、無秩序状態を確実に達成するためには、加熱ステップを過大評価する必要がある。
これにより、温度の無制限な探索の可能性が阻まれる。
そこで、私たちは、取得関数を制限して、基準温度から温度を上昇させるだけにした。
実験温度が推定されたTNを大きく上回り、それ以上の情報が得られなくなるまで、測定プロセスが繰り返される。
図1にANDiE方式の構成図を示す。

FIG. 1.
: 自律型中性子回折装置ANDiE（Autonomous neutron diffraction explorer）のアルゴリズム図。
ANDiEは、ネール温度（TN）を発見し、その後の磁気構造の温度依存性の仮説検証を行うために、ライブの中性子回折実験を自律的に駆動する。
実線はオークリッジ国立研究所（ORNL）高フラックス同位体炉（HFIR）のHB-2Cビームラインの広角中性子回折計（WAND2）に実装したもので、破線は国立標準技術研究所（NIST）中性子研究センター（NCNR）のビームラインBT-4に追加実装した能動学習ループを示す。

米国国立標準技術研究所（NIST）中性子研究センター（NCNR）のBT-4ビームラインの点検出器とオークリッジ国立研究所（ORNL）高フラックス同位体炉（HFIR）のHB-2Cビームラインの広角中性子回折計（WAND2）の両方にANDiEが導入されました。
ANDiEは、人間の介入なしにこれらの装置を制御し、MnOとFe1.09Teの両方の粉末試料のTNを自律的に発見することに成功し、そのための温度ステップ数を従来のアドホックスケジュール実験と比較して約5分の1に削減することが可能になった。
さらに、ANDiEは仮説検証を行い、磁気構造の温度依存性に対して正しい物理モデル（1次、Ising型2次、Weiss型2次）を特定します。
そして、そのモデルを尤度の高い順に並べ、一番上のモデルを選択する。このように、ANDiEは中性子回折実験の効率を飛躍的に向上させることが可能であることを示した。

## II. MATERIALS AND METHODS

### A. Algorithm

For the isothermal inference, we used model for the diffraction intensity in the 2θ diffraction space range of interest constructed from two Pseudo–Voigt peaks and a constant background.
The full model for the diffraction intensity in 2θ-space is given by
Equation(1)
where ζMag and ζStruct are scaling factors, ωMag and ωStruct are the peak locations in 2θ-space, and αMag, γMag, αStruct, and γStruct are the Pseudo–Voigt peak shape parameters for the magnetic and structural peaks, respectively, and IBack is the background intensity.

As discussed Sec. III A, we use the Weiss equation to predict the temperature dependence of the local magnetic moments during the autonomous experiment.30
With the assumption that the diffraction peak shape does not strongly change over the course of the experiment, the maximum intensity of the magnetic diffraction peak is proportional to the integrated intensity.
The full Weiss model for the temperature dependence of the magnetic diffraction intensity is then given by
Equation(2a)
Equation(2b)
where root𝑚[] is the root finding operation of the expression in the square brackets with respect to m, m is the reduced magnetization, T is the temperature in kelvin, I(T) is the diffraction intensity, TN is the magnetic transition temperature, J is the quantum total angular momentum, M0 is a scaling parameter proportional to the maximum spontaneous magnetization, BJ(x) is the Brillouin function, and Bk is the background intensity.
Note that M0 in Eq. (2a) is a composite of the maximum spontaneous magnetization and the square root of the unknown proportionality constant between the diffraction intensity and the square of the magnetic moments.

To select the next temperature, temperature values are explored (with a step size of 0.5 K) to identify the next temperature where the confidence interval (CI) of the model exceeds a threshold relative the Poissonian-like uncertainty predicted by the mean of the model.
Limiting the active learning scheme to increasing temperature avoids any hysteretic effects.
Once the temperature is above the upper confidence bound of TN, the confidence interval of the model no longer depends on temperature (as the background is the only parameter left to fit).
Large temperature steps above TN are then taken.

Once the full data set is collected in the autonomous experiment, ANDiE performs a post-processing hypothesis testing to determine which of the models discussed in Sec. III is the most likely.
The first-order model is given by 
Equation(3)
where K is the intensity scaling constant and σ is the full-width at half maximum of the Gaussian convolution of the step-function, which is used to describe the width of the transition.
Finally, the Ising model is given by 
Equation(4)
where β is the critical exponent.
Note that Eq. (4) is only valid near TN, which we have used T > 0.5TN to enforce.

ANDiE performs the inference using each of the three models.
Note that the inference with the Ising model infers TN and, therefore, also the range in which the Ising model is valid.
To ensure a fair comparison of the log-likelihoods between each of the three models, once the Ising model inference is complete, we re-perform the inference of the first-order and Weiss models using only the data points that fall within the valid range of the Ising model.
If the Ising model is not the most likely given the data points within that range, we compare the first-order and Weiss models using the inference on the entire temperature range acquired by the autonomous experiment.

All parameters of each model are initialized with prior truncated normal distributions based on the physical limitations (e.g., TN cannot be negative) and estimates from experts.
ANDiE uses the DREAM sampler31 to perform the MCMC inference.
For each of the models (in both the isothermal and thermal inference), we use a Gaussian likelihood around the prospective curve to determine the probability of observing the data given the models.
The widths of these likelihood distributions are determined by model and the instrument uncertainties,32,33 which captures the highly heteroscedastic nature of these Poissonian-like processes.
The autonomous analysis of the neutron diffraction data shown here was enabled by data pipeline that automatically reduces neutron event data into spectra34 using the Mantid framework.35

The full algorithm was written in Python and implemented in a Jupyter notebook that analyzes the diffraction patterns, selects the next temperature, and communicates with data acquisition, without human intervention.
The ANDiE notebooks used during the autonomous experiments are available at https://github.com/usnistgov/ANDiE-v1_0.
The BUMPS library36 was used for the MCMC functions with the DREAM sampler.31
For the thermal inference, a numerical root-seeking algorithm from the sci-kit learn library37 is used to solve for the root of the Weiss equation at each step in the MCMC chain.

### B. Experimental set-up

Diffraction experiments were performed at the WAND2 HB-2C beamline at HFIR at ORNL using a wavelength of 1.4828 Å.
Initial algorithm development was using experiments performed at the BT-4 beamline at the NCNR at NIST.
The MnO powder was purchased from Sigma Aldrich (Cat. # 377201)*.
A description of the synthesis details of the Fe1.09Te powder sample can be found in Ref. 38.
Both MnO and Fe1.09Te powder samples were measured in Vanadium cans sealed under He-atmosphere.
To reach low temperatures, a top-loading closed cycle refrigerator with a variable temperature insert (VTI) with He-exchange gas was used.

## III. RESULTS AND DISCUSSION

### A. Autonomous discovery of magnetic transition behavior of MnO

In this first demonstration of an autonomous research neutron diffraction system, we initially consider the well-studied material MnO to ensure ANDiE can reproduce known results.
In Sec. III B, we consider the more challenging material Fe1.09Te, which, as ANDiE discovers, has a sharp first-order transition.
These studies demonstrate the robustness of ANDiE and future studies can, therefore, confidently use ANDiE to study materials where the magnetic transition behavior is unknown.
There are some materials where the magnetic propagation vectors—and therefore diffraction peak positions in 2θ-space—are strong functions of temperature.39,40
Additionally, the intensity of the magnetic contribution to the diffraction pattern is also a strong function of temperature, especially across the ordering temperature [compare Fig. 2(a) with Fig. 2(b)].
As a result, the isothermal model parameters can change dramatically as the temperature-dependence experiment progresses.
This motivates the need for a reliable algorithmic platform capable of capturing such diverse behavior.
Bayesian inference provides a robust, probabilistic method to describe the material at any one temperature and across temperatures.
Bayesian inference allows one to utilize prior knowledge to improve data analysis and prediction, and it provides a framework for uncertainty quantification and propagation.
In contrast to simpler methods such as least squares fitting, this Bayesian framework allows the parameters to be inferred from the data with more accurate uncertainties.
Data can be input with uncertainty bounds, and target parameters are output as probability distributions with expected value and uncertainty.
In particular, ANDiE uses MCMC-based Bayesian inference to extract the magnetic component from each isothermal diffraction measurement.
The use of MCMC inference for global optimization ensures high confidence in peak parameter determination despite the large range of potential parameters values.
MCMC is particularly well suited to avoiding the myriad of local minima present in diffraction data.41
In contrast, other optimization schemes, such as the Levenberg–Marquardt algorithm, can perform well only when the initialization is close to the global minimum and can diverge otherwise.
Additionally, MCMC allows us to encode prior physics knowledge such as the Poissonian-like counting statistics of the measured intensities, thereby accounting for the highly heteroscedastic nature of the intensity as a function of diffraction angle (and of temperature, as shown in the next paragraph).
Furthermore, MCMC prior estimates of the parameters can be included, i.e., nuclear peak positions from previous x-ray diffraction measurements, or information from previous reports in the literature.
The active learning process begins with a previously identified range of interest for 2θ.
For MnO, we started with the detector 2θ range of 28.0° to 37.0°, which includes the (111) nuclear peak and the nearby (32 12 12) magnetic peak.
ANDiE then infers probability distributions for the peak shape parameters including the locations, heights, half-widths at half maximum for both the Gaussian and Lorentzian components of Pseudo–Voigt peaks, as well as a background term.
Figure 2 shows the results of this inference at 5.0 K and at 129.5 K.

FIG. 2.
: Isothermal inference for MnO.
Isothermal inference was performed on MnO diffraction data in the range of interest at the experiment temperature (T) of (a) 5.0 K and (b) 129.5 K.
The magnetic confidence interval in orange shows the confidence interval of the magnetic component of the isothermal model.
Note how the magnetic peak parameters (32 12 12) near 32.18° changes between the temperatures.
The global optimization MCMC algorithm infers an appropriate profile despite the large changes.
Error bars on the measured data points in blue represent one standard deviation and are smaller than the symbol size.

ANDiE uses the inferred peak parameter distributions at each isothermal measurement to predict the temperature dependence of the diffraction's magnetic component.
The magnetic component of neutron diffraction intensity is related to the square of the magnetic moment.29
Several models can describe the temperature dependence.
In this work, we consider a first-order phase transition model and the Ising and Weiss second-order phase transition models.
The first-order phase transition model is an error function—a step function convolved with a narrow Gaussian function.
Because this model only has a non-zero slope near TN, predictions made by inference are not informative for selecting subsequent temperature steps, i.e., there is no indication that the experiment temperature is approaching TN until it is within a few kelvin (i.e., within the Gaussian convolution).
Therefore, even if the material being studied is suspected to have first-order transition behavior, using a first-order model is not appropriate during the autonomous experiment.
Instead, ANDiE uses a second-order model to make predictions during the autonomous experiment.
A Bayes factor test is then used once all the data has been collected to determine if the material exhibits first-order behavior.

The second-order phase transition models do have non-zero slopes far below TN and, therefore, can be used to predict an appropriate temperature step.
However, the Ising model is only valid near TN, in the range 0.5TN < T < TN,42 whereas the Weiss model is valid across the entire temperature range below TN.
Therefore, regardless of the material being studied and the suspected behavior, ANDiE uses the Weiss model during the autonomous experiment to drive the data acquisition, and then in post-processing it determines the most appropriate model with the Bayes factor.
Example curves of each of these models are shown in Fig. 3.

FIG. 3.
: Example models for thermal inference.
There are several models for the temperature dependence of the magnetic component of the neutron diffraction intensity.
The black curve shows the first-order model.
The Ising model is shown in red, where the dashed region is outside the range of validity of this model.
The Weiss model is shown in blue.

For these reasons—regardless of the material being studied—ANDiE uses the Weiss model to select the temperature steps of subsequent measurements, propagating knowledge from low to high temperatures.
This physics-informed approach has several advantages over a more generic power law fitting or a surrogate ML model such as a Gaussian process.
First, the Weiss model constrains ANDiE to only physically meaningful solutions, i.e., positive temperature, positive intensity, and monotonic temperature dependence.
Second, we demonstrate that with the Weiss model, ANDiE focuses measurements in the most informative regions at low temperature and surrounding the transition temperature.
Furthermore, ANDiE uses a minimum number of measurements to properly characterize the curvature and background outside these regions.
We found this to be true regardless of the actual materials behavior and using only a broad prior estimation of the TN.
For materials with second order transitions, the Weiss model is flexible enough to drive the data collection to the informative temperatures.
In the case of a truly abrupt discontinuous first-order transition, discovering such a step-function is a daunting task that can only be solved iteratively with several cooling and warming cycles and ANDiE could be implemented to autonomize these iterations.
However, if there is a perturbation from ideal first-order behavior (i.e., from short-range order or the like) as is the case for many materials, the flexibility of the Weiss model allows ANDiE to collect more data near TN.
In this way, ANDiE uses the Weiss model to discover TN (with enough data to determine the transition behavior) from a single warming cycle.

Because neutron diffraction intensity obeys Poissonian-like counting statistics, the process is highly heteroscedastic, meaning that the uncertainty in the signal is highly non-uniform across the search domains.
The uncertainty (as estimated by the standard deviation of a Gaussian distribution using the continuous approximation) of the diffraction intensity is related to the square root of the diffraction intensity.
Common acquisition functions do not account for heteroscedasticity and tend to over-emphasize regions of high intensity, unnecessarily acquiring more data in these regions.
ANDiE, therefore, compares the confidence interval of the model (a value dependent on the number of measurements) to the uncertainty predicted from the intensity extrapolation.
ANDiE increases the temperature until that ratio is above some threshold which we call the Bravery factor.
The model variance is a measure of how well known the intensity is at each temperature given the data that has been measured.
The predicted uncertainty is a measure of how much we should expect to know about the value of intensity if a measurement is performed at that temperature.
The ratio of these two values represents how informative that measurement will be to the model.
Setting the Bravery factor determines a threshold on this ratio, above which measurements are considered useful.
Temperatures where the ratio is below the Bravery factor can be safely skipped, and the experiment temperature can be increased until that threshold is reached.
The Bravery factor therefore represents the user's risk tolerance and can change depending on the purpose of the experiment.
If little is known about the material, a high Bravery Factor might be appropriate to explore the space quickly.
If, however, the goal of the experiment is to fine tune the measurement of TN, then a smaller Bravery Factor might be appropriate so as to only take small temperature steps (i.e., measurements that are only moderately informative are still useful).

Figure 4 shows how ANDiE performed for the autonomous discovery of TN of MnO.
ANDiE chooses small temperature steps in the beginning of the autonomous experiment as there are little data to infer the temperature dependence.
As more data are acquired, ANDiE takes larger temperature steps until it approaches TN.
Near TN, the steep slope of the model naturally causes wide confidence intervals of the inference, and more data are acquired in the region.
In this way, ANDiE skips uninformative temperatures and quickly converges on TN.
After 14 temperature steps, ANDiE inferred that the experiment temperature was above TN.
In this region, the selection of further data points is arbitrary, and several measurements were taken at 10 K steps.
The results of this inference at several stages are shown in Fig.4.
After 16 temperature steps, ANDiE reached the stopping criteria for the experiment.
ANDiE quickly converged on the most likely parameters.
As mentioned before, there is no universally agreed method for the traditional ad hoc scheduling, which is determined by the intuition of the researcher and is particularly difficult with small signals (i.e., from materials with small magnetic moments, small crystal samples or thin-film).
For the sake of comparison, an informed ad hoc schedule might take 0.5 K steps within 10 K of prior guess of TN, 2 K steps within 20 K of the prior guess of TN, and 5 K steps otherwise, for a total of 74 temperature steps.
ANDiE, therefore, reduces the number of temperature steps required for the experiment by a factor of ≈5.

FIG. 4.
: Thermal inference snapshots for MnO.
The thermal inference step was performed during the autonomous experiment for temperature dependence of the MnO magnetic (32 12 12) reflection using the Weiss model.
Results are shown as determined after (a) one measurement, (b) after 12 measurements near the Néel temperature (TN), and (c) after 16 measurements at the end of the autonomous experiment.
The mean of the posterior curves (MP) of the inference for each model is shown in red.
The best parameters (BP) for each model are shown in the black dashed curves, while the confidence intervals (CI) are shown as the gray envelope.
The vertical green line in each part shows the next temperature the algorithm selected to measure next.
The error bars on the measured data points are smaller than the makers shown in blue, and in (c) the confidence interval of the model is smaller than the linewidth of the mean of the posteriors.

After the autonomous experiment reaches the stopping criteria, ANDiE performs hypothesis testing to determine which of the models considered herein are more likely.
Since the Ising model is only valid near TN, it determines the range over which the model likelihoods will be compared.
Thus, ANDiE performs inference with the Ising model first.
Inference is then performed for the other models over the same temperature range.
If the Ising model is not the most likely over the appropriate range, then ANDiE compares the likelihoods of the first-order model and Weiss model over the entire data set.
Figure 5 shows the result of the Ising model inference, determined to be the most likely model, with an estimated TN of 120.81(56) K.
Table I summarizes the results for all models.
The uncertainties in the prediction of TN reflect the confidence of the model in that parameter given the data points.
These confidence intervals represent an uncertainty in the parameter only insofar as the model is physically applicable.
For example, in the case of MnO, the Weiss model is not likely physically meaningful.
This is, therefore, also true of uncertainty in TN as derived from the Weiss model for that sample.
The fact that the first-order and Weiss models have such unphysically low uncertainty in this prediction shows that adjusting TN further will not improve the fit to the data.
This is also reflected in the large negative log-likelihoods of these models showing that they are not appropriate for the data.
In contrast, the Ising model is appropriate, as evidenced by the higher log-likelihood.
Therefore, the confidence in the TN parameter from the Ising model inference is a good measure of the uncertainty in TN.
Following this, ANDiE concludes that MnO is an Ising-type antiferromagnet with a TN of 120.81(56) K, consistent with the literature.42–44

FIG. 5.
: Ising model hypothesis testing.
Ising model inference was performed on the temperature dependence of the MnO magnetic (32 12 12) reflection as performed the post-processing hypothesis testing step and determined to be the most likely model.
The error bars on the measured data points are smaller than the makers, and the confidence interval of the model is smaller than the linewidth of the mean of the posteriors.

TABLE I.
: Thermal inference results.
The predicted Néel temperature (TN) and model log-likelihoods are used in the post-processing hypothesis testing for the autonomous neutron diffraction study of MnO and Fe1.09Te with WAND2 at HB-2C at HFIR at ORNL.
Note that the Ising model is only valid near TN.
The uncertainty in the prediction of TN reflects the confidence of that model in that parameter given the data.
These confidence intervals are good measures of the uncertainty only when the models are physically appropriate—as evidenced by the higher log-likelihoods.
Note that uncertainty is presented in compact notation where (##) represents the uncertainty in the last two digits of the value.
Boldface denotes the results of the most likely model for each material as concluded by ANDiE.

### B. Autonomous Discovery of Magnetic Transition Behavior of Fe1.09Te

Having validated its effectiveness on determining the magnetic transition of MnO, a well-studied material with a second-order transition, ANDiE was then implemented on the more challenging Fe–Te system.
Fe1+xTe has complicated magnetic behavior as a function of the interstitial iron, i.e., x in the chemical formula.45
Below ≈11% interstitial Fe, there is a first-order phase transition to an antiferromagnetic phase.
TN of this transition in Fe1+xTe ranges from 70 K at x = 0% to 52 K at x = 11%.
The precise determination of TN for Fe1.09Te is a challenging task since abrupt step-like first-order transition could occur over a wide range of temperatures.
Indeed, Fig. 6(a) shows sharp this transition is in the diffraction intensity of the magnetic (12 0 12) reflection of Fe1.09Te at 69.436(55) K, as acquired by an ad hoc measurement schedule (carried out as a separate experiment after the autonomous run).
ANDiE, using the Weiss-type transition model as a prior, discovered this transition in only 14 measurements.
This is an improvement over the ad hoc schedule by a factor of ≈4.
A discussion of the time savings, computational considerations, as well as a video capturing the evolution of inference as the autonomous experiment was performed are available in the supplementary material.
We note that the code currently used to implement the ANDiE is developmental, and while it is robust enough to demonstrate the autonomous decision making, future work could use parallel computing to speed computation (see discussion of computational time in the supplementary material, Sec. II).
After the data are collected, ANDiE then performs the model comparison between the Ising, Weiss, and first-order models [as shown in Figs. 6(b)–6(f)].
It can be seen that while the Ising model [Fig. 6(d)] has reasonably good agreement with the data, the first-order model over the same temperature range [Fig. 6(f)] provides a better description of the behavior.
This is especially evident in the region near the TN.
This low intensity data have low uncertainty owing the Poisonian-like statistics.
As a result of propagating this measurement uncertainty through this Bayesian framework, a few counts deviation between the model and the measurement at low intensity is far less likely than a few counts deviation at high intensity.
Therefore, deviations between the model and the data are more heavily penalized at low intensity in the calculation of the model likelihood.
The summary of the log-likelihoods and predicted transition temperatures used for the model selection is presented in Table I.
Here, it is worth noting that, as was the case above, the confidence intervals from the models are a good measure of the uncertainty only when the model is physically appropriate.
Considering the data from full temperature range, the high log-likelihood of the first-order model indicates that this model is the most likely.
ANDiE, therefore, correctly inferred first-order transition behavior with TN of 68.58(16) K, which agrees with that measured by the ad hoc schedule.
This magnetic ordering behavior is similar to what is expected from earlier reports of the Fe1+xTe phase diagram.45
ANDiE is, therefore, able to discover the behavior of the magnetic order parameter in very few measurements even when the actual behavior is far from the prior estimation.

FIG. 6.
: Autonomous measurement and hypothesis testing vs ad hoc schedule.
The intensity of magnetic (12 0 12) reflection of Fe1.09Te as determined from the isothermal inference are shown in blue.
The ad hoc schedule is shown in (a).
The hypothesis testing was performed using the inference of the (b) Weiss, (c) first-order, and (d) Ising models, respectively, on the autonomously acquired measurements.
The (e) Weiss and (f) first-order models were re-trained on the data where experiment temperature greater than one half the estimate of the Néel temperature (TN) from the Ising inference (i.e., T > 0.5TN).
The mean of the posterior curves (MP) of the inference for each model is shown in red.
The best parameters (BP) for each model are shown in the black dashed curves, while the confidence intervals (CI) are shown as the gray envelope.
The Weiss model was used to drive the autonomous experiment.
The first-order model was determined to be the most likely for this transition of Fe1.09Te.
The error bars on the measured data points shown in blue are smaller than the makers, and the confidence interval of the model is smaller than the linewidth of the mean of the posteriors.

These results show that ANDiE is capable of autonomously discovering TN of a material and performing basic model selection in the first live autonomously driven neutron diffraction experiments.
This demonstration goes beyond a simple proof-of-concept by making a discovery of TN and transition behavior on a previously under-studied material.
Furthermore, we have demonstrated the efficacy of a single pass of ANDiE, which can accomplish the goal of discovering TN to less than a degree and perform simple model selection.
An accurate determination of critical exponents is beyond the scope of this prototyping work, but ANDiE could be generalized to accomplish that task.
Future work can include conditions for cooling back to base temperature for subsequent runs of ANDiE with updated priors in order to perform more difficult discovery tasks.

## IV. CONCLUSION

We have developed the autonomous neutron diffraction explorer (ANDiE), a system for controlling neutron diffraction experiments for the discovery of the magnetic transition temperature.
The system presented here provides a Bayesian approach to selecting the experiment temperatures which not only provides probabilistic predictions but also encodes the relevant physics to the problem at hand.
We have demonstrated the versatility of ANDiE, which is capable of discovering the magnetic transition temperature of material systems with differing magnetic behaviors despite always driving the acquisition of data with the Weiss model.
Even though this model might not be the expected behavior of the material being studied, it is useful in choosing the next temperatures to efficiently discover the transition temperature.
ANDiE can accelerate the data acquisition by reducing the number of temperature steps by nearly a factor of 5 and can subsequently perform hypothesis testing to determine the governing physical principles of the transition.
The hypothesis testing after the data is acquired correctly identified the Ising-type transition in MnO at 120.81(56) K.
Even in the more challenging case of Fe1.09Te with an abrupt step-like first-order transition, ANDiE was able to efficiently drive the experiment, requiring only 14 measurements to discover the first-order transition behavior at 68.52(16) K.
As currently implemented, ANDiE compares the likelihood between the three previously discussed models.
Extending hypothesis testing to additional user-determined models is a straightforward task.
Furthermore, the methods implemented in ANDiE can be easily expanded to a variety of neutron-based experiments.
We expect the experiment speedup to increase with the dimensionality of the experiment such as implementing the active learning in the diffraction angle space on point-detector instruments such as the BT-4 at NCNR for rapid search of diffraction peaks.
Similarly, using our active learning scheme in the applied magnetic field space would reduce the number of measurements for those experiments.
The approach used by ANDiE is further generalizable to other measurements (such as x-ray diffraction or functional property measurements) where a generally applicable physical model can be used to efficiently navigate costly experimental conditions.
The autonomous system presented here exemplifies the potential of rapid neutron scatting experiments for accelerating materials discovery.

## LICENCE

Article by A. McDannald, M. Frontzek, A. T. Savici, M. Doucet, E. E. Rodriguez, K. Meuse, J. Opsahl-Ong, D. Samarov, I. Takeuchi, W. Ratcliff, & A. G. Kusne, "On-the-fly autonomous control of neutron diffraction via physics-informed Bayesian active learning", _Applied Physics Reviews_, **9** 021408 (2022). [https://doi.org/10.1063/5.0082956](https://doi.org/10.1063/5.0082956) / Cited under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) / Translated by [Yuki Sakishita](https://y-saki26.github.io/pages/).
