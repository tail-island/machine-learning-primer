Ô½<
Ù
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
*
Erf
x"T
y"T"
Ttype:
2
¿
ExtractImagePatches
images"T
patches"T"
ksizes	list(int)(0"
strides	list(int)(0"
rates	list(int)(0"
Ttype:
2	
""
paddingstring:
SAMEVALID
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ç7

layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelayer_normalization/gamma

-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:*
dtype0

layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namelayer_normalization/beta

,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0

layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_1/gamma

/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:*
dtype0

layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_1/beta

.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:*
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0

layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_2/gamma

/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes
:*
dtype0

layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_2/beta

.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0

layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_3/gamma

/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma*
_output_shapes
:*
dtype0

layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_3/beta

.layer_normalization_3/beta/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta*
_output_shapes
:*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	*
dtype0
s
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
l
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes	
:*
dtype0
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes
:	*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes
:*
dtype0

layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_4/gamma

/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes
:*
dtype0

layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_4/beta

.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
_output_shapes
:*
dtype0
z
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_13/kernel
s
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes

:*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
z
dense_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_14/kernel
s
#dense_14/kernel/Read/ReadVariableOpReadVariableOpdense_14/kernel*
_output_shapes

:*
dtype0
r
dense_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_14/bias
k
!dense_14/bias/Read/ReadVariableOpReadVariableOpdense_14/bias*
_output_shapes
:*
dtype0
z
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_15/kernel
s
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes

:*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes
:*
dtype0
z
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_16/kernel
s
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes

:*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:*
dtype0

layer_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namelayer_normalization_5/gamma

/layer_normalization_5/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_5/gamma*
_output_shapes
:*
dtype0

layer_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namelayer_normalization_5/beta

.layer_normalization_5/beta/Read/ReadVariableOpReadVariableOplayer_normalization_5/beta*
_output_shapes
:*
dtype0
{
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_17/kernel
t
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes
:	*
dtype0
s
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
l
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes	
:*
dtype0
{
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_18/kernel
t
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes
:	*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  @
¤
Const_1Const*"
_output_shapes
:1*
dtype0*à
valueÖBÓ1"Àu~¼³Ûª¼s×ì¼á(=­ñù¼S!)=IÐ5=:qK½Úò¼gaK=S;)î =h0<Í¼A½S1=v§¼ZØ7½ 
¼\<Z¨½q ½1=<=°»@?)<¢<ó7=[= ]m¼B=Ðýê»06Y;¶`< ;<gÇ1= }»wÊ =º¡Á¼Í¯<Æ(ÿ<ô9¼Fp½ÍÍ0=º6&½mÚ½ÆW< =2Ôï<ÚÑ½p,>½öÞ*½´À`<5¼gË3=g(=¦ø¼@¾A½Lín¼mÜá¼'Oø¼ômB¼XÇ¼ Uú;Ó=ç¡É¼Ø$¼ ¼Íû¼K½hbs¼£=cg=0Ut;gÖ¼ ¹¥:ÐÉ2» k»t÷g¼²Ç<V3½.äà<­|½báÌ<ªÛ7½S¸G=±'=@0<¿£¼©M¼ú¡½<@¼@µÆ: q\¼´.:¼:d½X¼G<Ý$=u½I=Ñ¼£7,½ÓÃµ¼ ^¸;ho¼3=qJ=ñ¼Å<ÓÄ1½&Ú½Q:­ÙÄ¼Ðú[;Lò¼óÚñ¼=Ùä=Iè*= G»tn¼¾ö¼9:=<ú½Ñ¤>=S@(½À@=]\=½}õ6=g!½õ»T¼³_=§Zù¼Ma¬¼@j¼0ú5»¼q== 3¼åâ;ñÛ<Úª<í°¼Ñ`==Æ¹û<R^<Üí<(T¼<±ï7=0W;m+)=®Cð<sí=fÎ¼x<Àîº tü¼F7<ÌM¼±ßC=­JÜ¼¡=p0½½ÿ½;»½7}I=G ¼ð0½¨ãn¼P¢Â;tÔ¼ùû¹mþ½Q:¼Ë-<Ð¤b;ØØ<Ðmë»sFô¼XT¼î¹á<i\¼`^)»X<<:º¼¦#¼ã*=dC=F3Ö< àP;'ºÁ¼(é»³b*=n.=ë0½J=ZÂL½RÃ×<3Êñ¼Rþí<º?§<ÌÖ<I9Ð
G;<À yºWþ<ô&¼g*À¼pý½ç¹E½l0=zÄ¤¼æ:=sõ(½ý¨2=j0=0¤»mÉÏ¼Át%=v>½rS<`è\»Zh½÷ô=½ZÍ¼:ÿ¼½=t½`Ó¼îo<î5ñ< 54½geî¼m;nò<Xë'<*ò¼=¹;F½-V=J±;²B¥<)`%=z^þ¼½Z½½Ñ ;ôa¼jM½'üî¼º¼ G» ½Ë<çk=¼<tÍ¼°.½ZA½z¡¤< Ðè¼ ;(¼Úª¼¨.<ûç¼M÷I½è·¼²®¼ÏZ<qê=æ¸»@ß<×=º%½@O)º±Ý=!Ð<Yt#= ½¹À.= `¤¹º.<ÇG3½WoH=ª©»`¨ü:ª=ú< F½fq¼<2 <Ø²U¼~å<ì7½¬¼ ±»(Z:¼ E½ÀG½ÒC½BCÖ<Àc3½0#Ô;½½:I<i<~@½t¼ÎÉ<ÐU-½Ó9G=óU0½gF@½D<ÆG=09à»Â;£Ã6½ô1G¼ Df;dÐ<p¼ÚË<P;ÿ»Xp'¼Qæ=F=@(½Ù£=wp=úß¾<b÷<îmÜ<¡Þ¼¢õà<lé¼ æ:D½Êc@½V4=,»¼103=4¸4¼fn¼£:½Z¸<rPÆ<f½h!ð»Ù= «A½ Y½ÁÙ>=íñ+=hç¼îÃ<4W<¼Á==ík>=QT=ÀC½µ7½_<
½z6½mö¼"è¨<@4?<l­¼É=õb<Pü»Ò¼ê<àî¼ÍE=t~< f%»@'<3µ¼ â¼ -; <J<=t´{<ª½²'<X_v¼°]»Pë%½`ZÆºp|L½mtú¼b9ð<@uô:½0½A×<Òàï<¦Â¼'i4½³Ï=Á<&½<Úø¼²
<Ãí=0é½;æ!½
Ô<hì» ¾·¼×Û>½zà<ØØ(<Àº³Iú¼0Ä»7õG=Z8õ<4¥½J¹¼ cß;Ú½<}]@½ÃVB½ÝA:½mc£¼°Ú»pü!½ÌÀD<¨;3¼pÜ½·½Ð¼ »:áÊ<-ô!=æ<< H1<9 =4K½O½ÐÌ;p½@þY¼-h¼Sæ)=`þ;Q¼,£¼ó½Î/Â<¦°½Æ<î;¸<¨:¼@rä¼ãH"½(+¼dü	½4»¼Ç­<½ÆsÇ<@É¯ºÒ < Í;ÐÖ>½û?<¶<Ó=Pk¤»äbY<sZ¹¼­×ñ¼Ü »g= Bý¼ØC{¼Ðà«;@ø<µÕ<Gé=bÝ;:S<Ú±Ò¼Óp ½w©2½@> ¼7¼:Ô¼p=Íc3=30=g[H½1!=*½3WD=óÜ ¼°!ß;Ý,¼Ú¨¡¼=Ý>=->½î{ô<è`Ü»ó&þ¼('b¼úÄ?½¨Lj¼·fG='Î= ìÍ¸Xp<0VÐ»í6½!½Ñ<è½¼41#¼Ý?*½ `;ÀEÐ¼-À< 	Ð»n¼Z¶¼Sn0½M5½0­t»ØB<äe<9:½£Y=f	<³cH=PX»X½`ù0;çlÚ¼3À¼Pú»ºÖ¾<$Æ¼Ó5=ÝÅ¼°H½ÁI= ±²:¼; »÷%:=#É,½É³=sQ=×J½W«"=¢=r<V½4}¼ >¹S'Ë¼´¨=¼  ü·Á´¼"Æý<æä¡< «h» Ù-½'Æ¼Gí/=Y|J=a'=ÉH=¨\<ûD=d	¼]>=41¼) =Úæ<@Ç¼Úø¼b¸<µ<áÀI=X¡¼Â%=:<p(<½íþ!=âÔ<¨
¼ ÌG;P»LdX¼rAÑ<èHÞ»h)²;(m¼â£<m±>=@Í:£¼¼`æ:½`øH;3E=%ä»:½tï¼yx=¨Í8<Õ?<Ñ;ç<QMA=aB=ÀÞæº`¼-¶.½£¥@=cy(½Z¯<Èë¼hf@<s>Ï¼ý´ ½ªç½4]c¼Àzí¼X£K¼í%½pu¼¿#=Ç
=	 
=ºxÿ<°Ùö;Á*+=v½qÆ=ýª==ç¼ø÷;×÷9=M¥¼è-< hv¸tjH<ÆË½£=(Ùa¼X]¼!=Zè¼=ÇK=ÚD5½3rÑ¼=Ó½#&=Uü;:6õ¼éo1=-d=½M*ò¼zÖ<À¥C<ÖE½X©d¼bÄß<ñÓH=4¯h¼ÐÙJ½§4= ½Iû =÷¹=9n=Ê¼@¨¼Òñ¼¹«6=S:¼Ã@=§[.=eö< .½§½&§Ò<`ø­:óþ½hÚí»L s<Æëõ<C(=Ø	1<«2=á{-=X,¼ íO;Àí­¼è;¼Îu¹<òÂ¸<÷;0²2½aN<} ½ ª'<
¸¼R¾<mØ½½Ñ@=tÉ]<eÎ<Ð,=@é¼æà¼L«<Ý:>½èÂ¼;M=Ç¥ú¼]áC½9®¼3uC½º¼Ñ¼)Ê/=PÃ¾»´U¼`«¼_z¼ý=LÖj¼ÚJ¡¼.i<p¶¼1I<=³ùÓ¼6&½ ³9i¼w¨=UW<ö¼¦"Ï<P<gn>=R2Ò<Ð¬;ô£e<Ýâ½P;ð*½='è»<L²¼«'½×Ú=0O½°»çI=NÉ¾<½Q ½Æë³<ÎIÂ<´)<I =à7í¼°î>½0AÐ;&¼´dp¼9Ä=z B½

NoOpNoOp
÷û
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*¯û
value¤ûB û Bû
¸
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-3
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer_with_weights-4
layer-24
layer-25
layer-26
layer_with_weights-5
layer-27
layer_with_weights-6
layer-28
layer-29
layer_with_weights-7
layer-30
 layer-31
!layer-32
"layer_with_weights-8
"layer-33
#layer_with_weights-9
#layer-34
$layer_with_weights-10
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer-43
-layer_with_weights-11
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer-49
3layer-50
4layer_with_weights-12
4layer-51
5layer-52
6layer-53
7layer_with_weights-13
7layer-54
8layer_with_weights-14
8layer-55
9layer-56
:layer_with_weights-15
:layer-57
;layer-58
<layer-59
=layer_with_weights-16
=layer-60
>layer_with_weights-17
>layer-61
?layer_with_weights-18
?layer-62
@layer-63
Alayer-64
Blayer-65
Clayer-66
Dlayer-67
Elayer-68
Flayer-69
Glayer-70
Hlayer_with_weights-19
Hlayer-71
Ilayer-72
Jlayer-73
Klayer-74
Llayer-75
Mlayer-76
Nlayer-77
Olayer_with_weights-20
Olayer-78
Player-79
Qlayer-80
Rlayer_with_weights-21
Rlayer-81
Slayer_with_weights-22
Slayer-82
Tlayer-83
Ulayer_with_weights-23
Ulayer-84
Vlayer-85
Wlayer-86
Xlayer-87
Ylayer_with_weights-24
Ylayer-88
Zlayer-89
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_default_save_signature
b
signatures*
* 

c	keras_api* 

d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 

j	keras_api* 

k	keras_api* 
¥
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p_random_generator
q__call__
*r&call_and_return_all_conditional_losses* 
¯
saxis
	tgamma
ubeta
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses*
ª

|kernel
}bias
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	keras_api* 

	keras_api* 

	keras_api* 

	keras_api* 

	keras_api* 

	keras_api* 

	keras_api* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses*

¡	keras_api* 

¢	keras_api* 

£	keras_api* 

¤	keras_api* 

¥	keras_api* 

¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses* 
®
¬kernel
	­bias
®	variables
¯trainable_variables
°regularization_losses
±	keras_api
²__call__
+³&call_and_return_all_conditional_losses*
¬
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸_random_generator
¹__call__
+º&call_and_return_all_conditional_losses* 

»	keras_api* 
¸
	¼axis

½gamma
	¾beta
¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses*
®
Åkernel
	Æbias
Ç	variables
Ètrainable_variables
Éregularization_losses
Ê	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses*

Í	keras_api* 
®
Îkernel
	Ïbias
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses*
¬
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
Ú_random_generator
Û__call__
+Ü&call_and_return_all_conditional_losses* 

Ý	keras_api* 
¸
	Þaxis

ßgamma
	àbeta
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
å__call__
+æ&call_and_return_all_conditional_losses*
®
çkernel
	èbias
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
í__call__
+î&call_and_return_all_conditional_losses*
®
ïkernel
	ðbias
ñ	variables
òtrainable_variables
óregularization_losses
ô	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses*

÷	variables
øtrainable_variables
ùregularization_losses
ú	keras_api
û__call__
+ü&call_and_return_all_conditional_losses* 

ý	keras_api* 

þ	keras_api* 

ÿ	keras_api* 

	keras_api* 

	keras_api* 

	keras_api* 

	keras_api* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

	keras_api* 

	keras_api* 

	keras_api* 

	keras_api* 

	keras_api* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£_random_generator
¤__call__
+¥&call_and_return_all_conditional_losses* 

¦	keras_api* 
¸
	§axis

¨gamma
	©beta
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses*
®
°kernel
	±bias
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses*

¸	keras_api* 
®
¹kernel
	ºbias
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
¿__call__
+À&call_and_return_all_conditional_losses*
¬
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å_random_generator
Æ__call__
+Ç&call_and_return_all_conditional_losses* 

È	keras_api* 
¸
	Éaxis

Êgamma
	Ëbeta
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses*
®
Òkernel
	Óbias
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses*
®
Úkernel
	Ûbias
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
à__call__
+á&call_and_return_all_conditional_losses*

â	variables
ãtrainable_variables
äregularization_losses
å	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses* 

è	keras_api* 

é	keras_api* 

ê	keras_api* 

ë	keras_api* 

ì	keras_api* 

í	keras_api* 

î	keras_api* 
®
ïkernel
	ðbias
ñ	variables
òtrainable_variables
óregularization_losses
ô	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses*

÷	keras_api* 

ø	keras_api* 

ù	keras_api* 

ú	keras_api* 

û	keras_api* 

ü	variables
ýtrainable_variables
þregularization_losses
ÿ	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
¬
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses* 

	keras_api* 
¸
	axis

gamma
	beta
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
®
kernel
	bias
	variables
trainable_variables
regularization_losses
 	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses*

£	keras_api* 
®
¤kernel
	¥bias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses*
¬
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°_random_generator
±__call__
+²&call_and_return_all_conditional_losses* 

³	keras_api* 

´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses* 
®
ºkernel
	»bias
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
À__call__
+Á&call_and_return_all_conditional_losses*

Â	variables
Ãtrainable_variables
Äregularization_losses
Å	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses* 
¸
t0
u1
|2
}3
4
5
6
7
¬8
­9
½10
¾11
Å12
Æ13
Î14
Ï15
ß16
à17
ç18
è19
ï20
ð21
22
23
24
25
¨26
©27
°28
±29
¹30
º31
Ê32
Ë33
Ò34
Ó35
Ú36
Û37
ï38
ð39
40
41
42
43
44
45
¤46
¥47
º48
»49*
¸
t0
u1
|2
}3
4
5
6
7
¬8
­9
½10
¾11
Å12
Æ13
Î14
Ï15
ß16
à17
ç18
è19
ï20
ð21
22
23
24
25
¨26
©27
°28
±29
¹30
º31
Ê32
Ë33
Ò34
Ó35
Ú36
Û37
ï38
ð39
40
41
42
43
44
45
¤46
¥47
º48
»49*
* 
µ
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
a_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 

Íserving_default* 
* 
* 
* 
* 

Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 

Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
l	variables
mtrainable_variables
nregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 
* 
* 
* 
* 
hb
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*

t0
u1*

t0
u1*
* 

Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

|0
}1*

|0
}1*
* 

Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

¬0
­1*

¬0
­1*
* 

önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
®	variables
¯trainable_variables
°regularization_losses
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
jd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*

½0
¾1*

½0
¾1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

Å0
Æ1*

Å0
Æ1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ç	variables
Ètrainable_variables
Éregularization_losses
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses*
* 
* 
* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

Î0
Ï1*

Î0
Ï1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ö	variables
×trainable_variables
Øregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
jd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*

ß0
à1*

ß0
à1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

ç0
è1*

ç0
è1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
é	variables
êtrainable_variables
ëregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_8/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_8/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

ï0
ð1*

ï0
ð1*
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
ñ	variables
òtrainable_variables
óregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
÷	variables
øtrainable_variables
ùregularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
_Y
VARIABLE_VALUEdense_9/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_9/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_10/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_10/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
	variables
 trainable_variables
¡regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
ke
VARIABLE_VALUElayer_normalization_3/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_3/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE*

¨0
©1*

¨0
©1*
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEdense_11/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_11/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

°0
±1*

°0
±1*
* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEdense_12/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_12/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

¹0
º1*

¹0
º1*
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
»	variables
¼trainable_variables
½regularization_losses
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
ke
VARIABLE_VALUElayer_normalization_4/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_4/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*

Ê0
Ë1*

Ê0
Ë1*
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEdense_13/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_13/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ò0
Ó1*

Ò0
Ó1*
* 

Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEdense_14/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_14/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ú0
Û1*

Ú0
Û1*
* 

Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
Ü	variables
Ýtrainable_variables
Þregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
â	variables
ãtrainable_variables
äregularization_losses
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
`Z
VARIABLE_VALUEdense_15/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_15/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*

ï0
ð1*

ï0
ð1*
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
ñ	variables
òtrainable_variables
óregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
ü	variables
ýtrainable_variables
þregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEdense_16/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_16/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
ke
VARIABLE_VALUElayer_normalization_5/gamma6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUElayer_normalization_5/beta5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
`Z
VARIABLE_VALUEdense_17/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_17/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses*
* 
* 
* 
`Z
VARIABLE_VALUEdense_18/kernel7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_18/bias5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUE*

¤0
¥1*

¤0
¥1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¬	variables
­trainable_variables
®regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEdense/kernel7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUE
dense/bias5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUE*

º0
»1*

º0
»1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¼	variables
½trainable_variables
¾regularization_losses
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Â	variables
Ãtrainable_variables
Äregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses* 
* 
* 
* 
Ê
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
W86
X87
Y88
Z89*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

total

count
	variables
 	keras_api*
M

¡total

¢count
£
_fn_kwargs
¤	variables
¥	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¡0
¢1*

¤	variables*

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
Ä

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1ConstConst_1layer_normalization/gammalayer_normalization/betadense_2/kerneldense_2/biasdense_1/kerneldense_1/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biaslayer_normalization_1/gammalayer_normalization_1/betadense_5/kerneldense_5/biasdense_6/kerneldense_6/biaslayer_normalization_2/gammalayer_normalization_2/betadense_8/kerneldense_8/biasdense_7/kerneldense_7/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biaslayer_normalization_3/gammalayer_normalization_3/betadense_11/kerneldense_11/biasdense_12/kerneldense_12/biaslayer_normalization_4/gammalayer_normalization_4/betadense_14/kerneldense_14/biasdense_13/kerneldense_13/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biaslayer_normalization_5/gammalayer_normalization_5/betadense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense/kernel
dense/bias*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_78792
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ç
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp/layer_normalization_3/gamma/Read/ReadVariableOp.layer_normalization_3/beta/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp/layer_normalization_4/gamma/Read/ReadVariableOp.layer_normalization_4/beta/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp#dense_14/kernel/Read/ReadVariableOp!dense_14/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp/layer_normalization_5/gamma/Read/ReadVariableOp.layer_normalization_5/beta/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst_2*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_80225


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biaslayer_normalization_1/gammalayer_normalization_1/betadense_5/kerneldense_5/biasdense_6/kerneldense_6/biaslayer_normalization_2/gammalayer_normalization_2/betadense_7/kerneldense_7/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biaslayer_normalization_3/gammalayer_normalization_3/betadense_11/kerneldense_11/biasdense_12/kerneldense_12/biaslayer_normalization_4/gammalayer_normalization_4/betadense_13/kerneldense_13/biasdense_14/kerneldense_14/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biaslayer_normalization_5/gammalayer_normalization_5/betadense_17/kerneldense_17/biasdense_18/kerneldense_18/biasdense/kernel
dense/biastotalcounttotal_1count_1*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_80397ù3
¸
E
)__inference_reshape_4_layer_call_fn_79396

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_74616d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ê
ú
C__inference_dense_10_layer_call_and_return_conditional_losses_79448

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ü
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
´
C
'__inference_reshape_layer_call_fn_78797

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_74069d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å
`
B__inference_dropout_layer_call_and_return_conditional_losses_74080

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_3_layer_call_and_return_conditional_losses_75498

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_7_layer_call_and_return_conditional_losses_74524

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
°
E
)__inference_dropout_1_layer_call_fn_79066

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_74311d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ê
ú
C__inference_dense_15_layer_call_and_return_conditional_losses_74935

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

b
)__inference_dropout_4_layer_call_fn_79594

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_75445s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ122
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
×

(__inference_dense_17_layer_call_fn_79902

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_75068t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ó
ü
C__inference_dense_11_layer_call_and_return_conditional_losses_79545

inputs4
!tensordot_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_74427

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
À

%__inference_dense_layer_call_fn_80018

inputs
unknown:	

	unknown_0:

identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75144o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_75007

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
°
E
)__inference_dropout_5_layer_call_fn_79840

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_75007d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ê
ú
C__inference_dense_16_layer_call_and_return_conditional_losses_79835

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_79212

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

`
D__inference_reshape_6_layer_call_and_return_conditional_losses_79796

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ç	
ò
@__inference_dense_layer_call_and_return_conditional_losses_75144

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó
ü
C__inference_dense_17_layer_call_and_return_conditional_losses_79932

inputs4
!tensordot_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ñ

'__inference_dense_9_layer_call_fn_79361

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_74587s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¸
E
)__inference_reshape_1_layer_call_fn_78951

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
°
E
)__inference_dropout_2_layer_call_fn_79202

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_74427d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
é

5__inference_layer_normalization_2_layer_call_fn_79233

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_74452s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

b
)__inference_dropout_1_layer_call_fn_79071

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_75636s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ122
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ó

(__inference_dense_16_layer_call_fn_79805

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_74996s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_78837

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ñ

'__inference_dense_8_layer_call_fn_79303

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_74488s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¼
^
B__inference_flatten_layer_call_and_return_conditional_losses_75132

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ëµ
õ
@__inference_model_layer_call_and_return_conditional_losses_76786
input_1
tf_math_multiply_mul_y 
tf___operators___add_addv2_y'
layer_normalization_76553:'
layer_normalization_76555:
dense_2_76558:
dense_2_76560:
dense_1_76563:
dense_1_76565:
dense_3_76580:
dense_3_76582:
dense_4_76596:
dense_4_76598:)
layer_normalization_1_76603:)
layer_normalization_1_76605: 
dense_5_76608:	
dense_5_76610:	 
dense_6_76621:	
dense_6_76623:)
layer_normalization_2_76628:)
layer_normalization_2_76630:
dense_8_76633:
dense_8_76635:
dense_7_76638:
dense_7_76640:
dense_9_76655:
dense_9_76657: 
dense_10_76671:
dense_10_76673:)
layer_normalization_3_76678:)
layer_normalization_3_76680:!
dense_11_76683:	
dense_11_76685:	!
dense_12_76696:	
dense_12_76698:)
layer_normalization_4_76703:)
layer_normalization_4_76705: 
dense_14_76708:
dense_14_76710: 
dense_13_76713:
dense_13_76715: 
dense_15_76730:
dense_15_76732: 
dense_16_76746:
dense_16_76748:)
layer_normalization_5_76753:)
layer_normalization_5_76755:!
dense_17_76758:	
dense_17_76760:	!
dense_18_76771:	
dense_18_76773:
dense_76779:	

dense_76781:

identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢!dropout_6/StatefulPartitionedCall¢+layer_normalization/StatefulPartitionedCall¢-layer_normalization_1/StatefulPartitionedCall¢-layer_normalization_2/StatefulPartitionedCall¢-layer_normalization_3/StatefulPartitionedCall¢-layer_normalization_4/StatefulPartitionedCall¢-layer_normalization_5/StatefulPartitionedCallÖ
,tf.image.extract_patches/ExtractImagePatchesExtractImagePatchesinput_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksizes
*
paddingVALID*
rates
*
strides
ì
reshape/PartitionedCallPartitionedCall6tf.image.extract_patches/ExtractImagePatches:patches:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_74069
tf.math.multiply/MulMul reshape/PartitionedCall:output:0tf_math_multiply_mul_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ä
dropout/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_75721Â
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0layer_normalization_76553layer_normalization_76555*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_74104
dense_2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_2_76558dense_2_76560*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_74140
dense_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_1_76563dense_1_76565*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_74176æ
reshape_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196è
reshape_1/PartitionedCall_1PartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_1/transpose	Transpose"reshape_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.compat.v1.shape/ShapeShape&tf.compat.v1.transpose_1/transpose:y:0*
T0*
_output_shapes
:
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
tf.cast/CastCast/tf.__operators__.getitem/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ½
 tf.compat.v1.transpose/transpose	Transpose$reshape_1/PartitionedCall_1:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_3/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_3_76580dense_3_76582*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_74239½
tf.linalg.matmul/MatMulBatchMatMulV2$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(L
tf.math.sqrt/SqrtSqrttf.cast/Cast:y:0*
T0*
_output_shapes
: è
reshape_1/PartitionedCall_2PartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196
tf.math.truediv/truedivRealDiv tf.linalg.matmul/MatMul:output:0tf.math.sqrt/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11w
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_2/transpose	Transpose$reshape_1/PartitionedCall_2:output:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
tf.linalg.matmul_1/MatMulBatchMatMulV2tf.nn.softmax/Softmax:softmax:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_3/transpose	Transpose"tf.linalg.matmul_1/MatMul:output:00tf.compat.v1.transpose_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1à
reshape_2/PartitionedCallPartitionedCall&tf.compat.v1.transpose_3/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_74268
dense_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0dense_4_76596dense_4_76598*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_74300
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_75636½
tf.__operators__.add_1/AddV2AddV2*dropout_1/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_1_76603layer_normalization_1_76605*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_74336¡
dense_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_5_76608dense_5_76610*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_74372Z
tf.nn.gelu/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
tf.nn.gelu/Gelu/mulMultf.nn.gelu/Gelu/mul/x:output:0(dense_5/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1[
tf.nn.gelu/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?¤
tf.nn.gelu/Gelu/truedivRealDiv(dense_5/StatefulPartitionedCall:output:0tf.nn.gelu/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.nn.gelu/Gelu/ErfErftf.nn.gelu/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
tf.nn.gelu/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu/Gelu/addAddV2tf.nn.gelu/Gelu/add/x:output:0tf.nn.gelu/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu/Gelu/mul_1Multf.nn.gelu/Gelu/mul:z:0tf.nn.gelu/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_6/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu/Gelu/mul_1:z:0dense_6_76621dense_6_76623*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_74416
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_75583¿
tf.__operators__.add_2/AddV2AddV2*dropout_2/StatefulPartitionedCall:output:06layer_normalization_1/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_2_76628layer_normalization_2_76630*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_74452 
dense_8/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_8_76633dense_8_76635*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_74488 
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_7_76638dense_7_76640*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_74524æ
reshape_3/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544è
reshape_3/PartitionedCall_1PartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544
'tf.compat.v1.transpose_5/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_5/transpose	Transpose"reshape_3/PartitionedCall:output:00tf.compat.v1.transpose_5/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_1/ShapeShape&tf.compat.v1.transpose_5/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_1/CastCast1tf.__operators__.getitem_1/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_4/transpose	Transpose$reshape_3/PartitionedCall_1:output:00tf.compat.v1.transpose_4/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1 
dense_9/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_9_76655dense_9_76657*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_74587Á
tf.linalg.matmul_2/MatMulBatchMatMulV2&tf.compat.v1.transpose_4/transpose:y:0&tf.compat.v1.transpose_5/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_1/SqrtSqrttf.cast_1/Cast:y:0*
T0*
_output_shapes
: è
reshape_3/PartitionedCall_2PartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544
tf.math.truediv_1/truedivRealDiv"tf.linalg.matmul_2/MatMul:output:0tf.math.sqrt_1/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_6/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_6/transpose	Transpose$reshape_3/PartitionedCall_2:output:00tf.compat.v1.transpose_6/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¯
tf.linalg.matmul_3/MatMulBatchMatMulV2!tf.nn.softmax_1/Softmax:softmax:0&tf.compat.v1.transpose_6/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_7/transpose	Transpose"tf.linalg.matmul_3/MatMul:output:00tf.compat.v1.transpose_7/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1à
reshape_4/PartitionedCallPartitionedCall&tf.compat.v1.transpose_7/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_74616
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0dense_10_76671dense_10_76673*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_74648
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_75498¿
tf.__operators__.add_3/AddV2AddV2*dropout_3/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_3_76678layer_normalization_3_76680*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_74684¥
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_11_76683dense_11_76685*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_74720\
tf.nn.gelu_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
tf.nn.gelu_1/Gelu/mulMul tf.nn.gelu_1/Gelu/mul/x:output:0)dense_11/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?©
tf.nn.gelu_1/Gelu/truedivRealDiv)dense_11/StatefulPartitionedCall:output:0!tf.nn.gelu_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_1/Gelu/ErfErftf.nn.gelu_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_1/Gelu/addAddV2 tf.nn.gelu_1/Gelu/add/x:output:0tf.nn.gelu_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_1/Gelu/mul_1Multf.nn.gelu_1/Gelu/mul:z:0tf.nn.gelu_1/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_12/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu_1/Gelu/mul_1:z:0dense_12_76696dense_12_76698*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_74764
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_75445¿
tf.__operators__.add_4/AddV2AddV2*dropout_4/StatefulPartitionedCall:output:06layer_normalization_3/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_4_76703layer_normalization_4_76705*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_74800¤
 dense_14/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_14_76708dense_14_76710*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_74836¤
 dense_13/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_13_76713dense_13_76715*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_74872ç
reshape_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892é
reshape_5/PartitionedCall_1PartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892
'tf.compat.v1.transpose_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_9/transpose	Transpose"reshape_5/PartitionedCall:output:00tf.compat.v1.transpose_9/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_2/ShapeShape&tf.compat.v1.transpose_9/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_2/strided_sliceStridedSlice#tf.compat.v1.shape_2/Shape:output:07tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_2/CastCast1tf.__operators__.getitem_2/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_8/transpose	Transpose$reshape_5/PartitionedCall_1:output:00tf.compat.v1.transpose_8/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¤
 dense_15/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_15_76730dense_15_76732*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_74935Á
tf.linalg.matmul_4/MatMulBatchMatMulV2&tf.compat.v1.transpose_8/transpose:y:0&tf.compat.v1.transpose_9/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_2/SqrtSqrttf.cast_2/Cast:y:0*
T0*
_output_shapes
: é
reshape_5/PartitionedCall_2PartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892
tf.math.truediv_2/truedivRealDiv"tf.linalg.matmul_4/MatMul:output:0tf.math.sqrt_2/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_2/SoftmaxSoftmaxtf.math.truediv_2/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
(tf.compat.v1.transpose_10/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ã
#tf.compat.v1.transpose_10/transpose	Transpose$reshape_5/PartitionedCall_2:output:01tf.compat.v1.transpose_10/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1°
tf.linalg.matmul_5/MatMulBatchMatMulV2!tf.nn.softmax_2/Softmax:softmax:0'tf.compat.v1.transpose_10/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
(tf.compat.v1.transpose_11/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
#tf.compat.v1.transpose_11/transpose	Transpose"tf.linalg.matmul_5/MatMul:output:01tf.compat.v1.transpose_11/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1á
reshape_6/PartitionedCallPartitionedCall'tf.compat.v1.transpose_11/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_6_layer_call_and_return_conditional_losses_74964
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0dense_16_76746dense_16_76748*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_74996
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_75360¿
tf.__operators__.add_5/AddV2AddV2*dropout_5/StatefulPartitionedCall:output:06layer_normalization_4/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_5_76753layer_normalization_5_76755*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_75032¥
 dense_17/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0dense_17_76758dense_17_76760*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_75068\
tf.nn.gelu_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
tf.nn.gelu_2/Gelu/mulMul tf.nn.gelu_2/Gelu/mul/x:output:0)dense_17/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?©
tf.nn.gelu_2/Gelu/truedivRealDiv)dense_17/StatefulPartitionedCall:output:0!tf.nn.gelu_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_2/Gelu/ErfErftf.nn.gelu_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_2/Gelu/addAddV2 tf.nn.gelu_2/Gelu/add/x:output:0tf.nn.gelu_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_2/Gelu/mul_1Multf.nn.gelu_2/Gelu/mul:z:0tf.nn.gelu_2/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu_2/Gelu/mul_1:z:0dense_18_76771dense_18_76773*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_75112
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_75307¿
tf.__operators__.add_6/AddV2AddV2*dropout_6/StatefulPartitionedCall:output:06layer_normalization_5/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ó
flatten/PartitionedCallPartitionedCall tf.__operators__.add_6/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75132þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_76779dense_76781*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75144Ø
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_75155o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ë	
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: :($
"
_output_shapes
:1
Ñ

'__inference_dense_4_layer_call_fn_79031

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_74300s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_79893

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ñ

'__inference_dense_7_layer_call_fn_79264

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_74524s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_2_layer_call_and_return_conditional_losses_79224

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ü
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_79352

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_79255

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ó

(__inference_dense_14_layer_call_fn_79690

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_74836s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ê
ú
C__inference_dense_14_layer_call_and_return_conditional_losses_79720

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_3_layer_call_and_return_conditional_losses_74239

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
é

5__inference_layer_normalization_4_layer_call_fn_79620

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_74800s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_74311

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Â
ò,
 __inference__wrapped_model_74048
input_1 
model_tf_math_multiply_mul_y&
"model_tf___operators___add_addv2_yM
?model_layer_normalization_batchnorm_mul_readvariableop_resource:I
;model_layer_normalization_batchnorm_readvariableop_resource:A
/model_dense_2_tensordot_readvariableop_resource:;
-model_dense_2_biasadd_readvariableop_resource:A
/model_dense_1_tensordot_readvariableop_resource:;
-model_dense_1_biasadd_readvariableop_resource:A
/model_dense_3_tensordot_readvariableop_resource:;
-model_dense_3_biasadd_readvariableop_resource:A
/model_dense_4_tensordot_readvariableop_resource:;
-model_dense_4_biasadd_readvariableop_resource:O
Amodel_layer_normalization_1_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_1_batchnorm_readvariableop_resource:B
/model_dense_5_tensordot_readvariableop_resource:	<
-model_dense_5_biasadd_readvariableop_resource:	B
/model_dense_6_tensordot_readvariableop_resource:	;
-model_dense_6_biasadd_readvariableop_resource:O
Amodel_layer_normalization_2_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_2_batchnorm_readvariableop_resource:A
/model_dense_8_tensordot_readvariableop_resource:;
-model_dense_8_biasadd_readvariableop_resource:A
/model_dense_7_tensordot_readvariableop_resource:;
-model_dense_7_biasadd_readvariableop_resource:A
/model_dense_9_tensordot_readvariableop_resource:;
-model_dense_9_biasadd_readvariableop_resource:B
0model_dense_10_tensordot_readvariableop_resource:<
.model_dense_10_biasadd_readvariableop_resource:O
Amodel_layer_normalization_3_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_3_batchnorm_readvariableop_resource:C
0model_dense_11_tensordot_readvariableop_resource:	=
.model_dense_11_biasadd_readvariableop_resource:	C
0model_dense_12_tensordot_readvariableop_resource:	<
.model_dense_12_biasadd_readvariableop_resource:O
Amodel_layer_normalization_4_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_4_batchnorm_readvariableop_resource:B
0model_dense_14_tensordot_readvariableop_resource:<
.model_dense_14_biasadd_readvariableop_resource:B
0model_dense_13_tensordot_readvariableop_resource:<
.model_dense_13_biasadd_readvariableop_resource:B
0model_dense_15_tensordot_readvariableop_resource:<
.model_dense_15_biasadd_readvariableop_resource:B
0model_dense_16_tensordot_readvariableop_resource:<
.model_dense_16_biasadd_readvariableop_resource:O
Amodel_layer_normalization_5_batchnorm_mul_readvariableop_resource:K
=model_layer_normalization_5_batchnorm_readvariableop_resource:C
0model_dense_17_tensordot_readvariableop_resource:	=
.model_dense_17_biasadd_readvariableop_resource:	C
0model_dense_18_tensordot_readvariableop_resource:	<
.model_dense_18_biasadd_readvariableop_resource:=
*model_dense_matmul_readvariableop_resource:	
9
+model_dense_biasadd_readvariableop_resource:

identity¢"model/dense/BiasAdd/ReadVariableOp¢!model/dense/MatMul/ReadVariableOp¢$model/dense_1/BiasAdd/ReadVariableOp¢&model/dense_1/Tensordot/ReadVariableOp¢%model/dense_10/BiasAdd/ReadVariableOp¢'model/dense_10/Tensordot/ReadVariableOp¢%model/dense_11/BiasAdd/ReadVariableOp¢'model/dense_11/Tensordot/ReadVariableOp¢%model/dense_12/BiasAdd/ReadVariableOp¢'model/dense_12/Tensordot/ReadVariableOp¢%model/dense_13/BiasAdd/ReadVariableOp¢'model/dense_13/Tensordot/ReadVariableOp¢%model/dense_14/BiasAdd/ReadVariableOp¢'model/dense_14/Tensordot/ReadVariableOp¢%model/dense_15/BiasAdd/ReadVariableOp¢'model/dense_15/Tensordot/ReadVariableOp¢%model/dense_16/BiasAdd/ReadVariableOp¢'model/dense_16/Tensordot/ReadVariableOp¢%model/dense_17/BiasAdd/ReadVariableOp¢'model/dense_17/Tensordot/ReadVariableOp¢%model/dense_18/BiasAdd/ReadVariableOp¢'model/dense_18/Tensordot/ReadVariableOp¢$model/dense_2/BiasAdd/ReadVariableOp¢&model/dense_2/Tensordot/ReadVariableOp¢$model/dense_3/BiasAdd/ReadVariableOp¢&model/dense_3/Tensordot/ReadVariableOp¢$model/dense_4/BiasAdd/ReadVariableOp¢&model/dense_4/Tensordot/ReadVariableOp¢$model/dense_5/BiasAdd/ReadVariableOp¢&model/dense_5/Tensordot/ReadVariableOp¢$model/dense_6/BiasAdd/ReadVariableOp¢&model/dense_6/Tensordot/ReadVariableOp¢$model/dense_7/BiasAdd/ReadVariableOp¢&model/dense_7/Tensordot/ReadVariableOp¢$model/dense_8/BiasAdd/ReadVariableOp¢&model/dense_8/Tensordot/ReadVariableOp¢$model/dense_9/BiasAdd/ReadVariableOp¢&model/dense_9/Tensordot/ReadVariableOp¢2model/layer_normalization/batchnorm/ReadVariableOp¢6model/layer_normalization/batchnorm/mul/ReadVariableOp¢4model/layer_normalization_1/batchnorm/ReadVariableOp¢8model/layer_normalization_1/batchnorm/mul/ReadVariableOp¢4model/layer_normalization_2/batchnorm/ReadVariableOp¢8model/layer_normalization_2/batchnorm/mul/ReadVariableOp¢4model/layer_normalization_3/batchnorm/ReadVariableOp¢8model/layer_normalization_3/batchnorm/mul/ReadVariableOp¢4model/layer_normalization_4/batchnorm/ReadVariableOp¢8model/layer_normalization_4/batchnorm/mul/ReadVariableOp¢4model/layer_normalization_5/batchnorm/ReadVariableOp¢8model/layer_normalization_5/batchnorm/mul/ReadVariableOpÜ
2model/tf.image.extract_patches/ExtractImagePatchesExtractImagePatchesinput_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksizes
*
paddingVALID*
rates
*
strides

model/reshape/ShapeShape<model/tf.image.extract_patches/ExtractImagePatches:patches:0*
T0*
_output_shapes
:k
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ_
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Ç
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:º
model/reshape/ReshapeReshape<model/tf.image.extract_patches/ExtractImagePatches:patches:0$model/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
model/tf.math.multiply/MulMulmodel/reshape/Reshape:output:0model_tf_math_multiply_mul_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
 model/tf.__operators__.add/AddV2AddV2model/tf.math.multiply/Mul:z:0"model_tf___operators___add_addv2_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
model/dropout/IdentityIdentity$model/tf.__operators__.add/AddV2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8model/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ù
&model/layer_normalization/moments/meanMeanmodel/dropout/Identity:output:0Amodel/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(¥
.model/layer_normalization/moments/StopGradientStopGradient/model/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ø
3model/layer_normalization/moments/SquaredDifferenceSquaredDifferencemodel/dropout/Identity:output:07model/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
<model/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ù
*model/layer_normalization/moments/varianceMean7model/layer_normalization/moments/SquaredDifference:z:0Emodel/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(n
)model/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ï
'model/layer_normalization/batchnorm/addAddV23model/layer_normalization/moments/variance:output:02model/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
)model/layer_normalization/batchnorm/RsqrtRsqrt+model/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1²
6model/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ó
'model/layer_normalization/batchnorm/mulMul-model/layer_normalization/batchnorm/Rsqrt:y:0>model/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1´
)model/layer_normalization/batchnorm/mul_1Mulmodel/dropout/Identity:output:0+model/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ä
)model/layer_normalization/batchnorm/mul_2Mul/model/layer_normalization/moments/mean:output:0+model/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2model/layer_normalization/batchnorm/ReadVariableOpReadVariableOp;model_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ï
'model/layer_normalization/batchnorm/subSub:model/layer_normalization/batchnorm/ReadVariableOp:value:0-model/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ä
)model/layer_normalization/batchnorm/add_1AddV2-model/layer_normalization/batchnorm/mul_1:z:0+model/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
&model/dense_2/Tensordot/ReadVariableOpReadVariableOp/model_dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0f
model/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       z
model/dense_2/Tensordot/ShapeShape-model/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
%model/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
 model/dense_2/Tensordot/GatherV2GatherV2&model/dense_2/Tensordot/Shape:output:0%model/dense_2/Tensordot/free:output:0.model/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
"model/dense_2/Tensordot/GatherV2_1GatherV2&model/dense_2/Tensordot/Shape:output:0%model/dense_2/Tensordot/axes:output:00model/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_2/Tensordot/ProdProd)model/dense_2/Tensordot/GatherV2:output:0&model/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_2/Tensordot/Prod_1Prod+model/dense_2/Tensordot/GatherV2_1:output:0(model/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
model/dense_2/Tensordot/concatConcatV2%model/dense_2/Tensordot/free:output:0%model/dense_2/Tensordot/axes:output:0,model/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:£
model/dense_2/Tensordot/stackPack%model/dense_2/Tensordot/Prod:output:0'model/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¼
!model/dense_2/Tensordot/transpose	Transpose-model/layer_normalization/batchnorm/add_1:z:0'model/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1´
model/dense_2/Tensordot/ReshapeReshape%model/dense_2/Tensordot/transpose:y:0&model/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
model/dense_2/Tensordot/MatMulMatMul(model/dense_2/Tensordot/Reshape:output:0.model/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
model/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
 model/dense_2/Tensordot/concat_1ConcatV2)model/dense_2/Tensordot/GatherV2:output:0(model/dense_2/Tensordot/Const_2:output:0.model/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:­
model/dense_2/TensordotReshape(model/dense_2/Tensordot/MatMul:product:0)model/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model/dense_2/BiasAddBiasAdd model/dense_2/Tensordot:output:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
&model/dense_1/Tensordot/ReadVariableOpReadVariableOp/model_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0f
model/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       z
model/dense_1/Tensordot/ShapeShape-model/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
%model/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
 model/dense_1/Tensordot/GatherV2GatherV2&model/dense_1/Tensordot/Shape:output:0%model/dense_1/Tensordot/free:output:0.model/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
"model/dense_1/Tensordot/GatherV2_1GatherV2&model/dense_1/Tensordot/Shape:output:0%model/dense_1/Tensordot/axes:output:00model/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_1/Tensordot/ProdProd)model/dense_1/Tensordot/GatherV2:output:0&model/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_1/Tensordot/Prod_1Prod+model/dense_1/Tensordot/GatherV2_1:output:0(model/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
model/dense_1/Tensordot/concatConcatV2%model/dense_1/Tensordot/free:output:0%model/dense_1/Tensordot/axes:output:0,model/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:£
model/dense_1/Tensordot/stackPack%model/dense_1/Tensordot/Prod:output:0'model/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¼
!model/dense_1/Tensordot/transpose	Transpose-model/layer_normalization/batchnorm/add_1:z:0'model/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1´
model/dense_1/Tensordot/ReshapeReshape%model/dense_1/Tensordot/transpose:y:0&model/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
model/dense_1/Tensordot/MatMulMatMul(model/dense_1/Tensordot/Reshape:output:0.model/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
model/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
 model/dense_1/Tensordot/concat_1ConcatV2)model/dense_1/Tensordot/GatherV2:output:0(model/dense_1/Tensordot/Const_2:output:0.model/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:­
model/dense_1/TensordotReshape(model/dense_1/Tensordot/MatMul:product:0)model/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model/dense_1/BiasAddBiasAdd model/dense_1/Tensordot:output:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
model/reshape_1/ShapeShapemodel/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:m
#model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
model/reshape_1/strided_sliceStridedSlicemodel/reshape_1/Shape:output:0,model/reshape_1/strided_slice/stack:output:0.model/reshape_1/strided_slice/stack_1:output:0.model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
model/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ù
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0(model/reshape_1/Reshape/shape/2:output:0(model/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¤
model/reshape_1/ReshapeReshapemodel/dense_2/BiasAdd:output:0&model/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1e
model/reshape_1/Shape_1Shapemodel/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:o
%model/reshape_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model/reshape_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model/reshape_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
model/reshape_1/strided_slice_1StridedSlice model/reshape_1/Shape_1:output:0.model/reshape_1/strided_slice_1/stack:output:00model/reshape_1/strided_slice_1/stack_1:output:00model/reshape_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!model/reshape_1/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!model/reshape_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model/reshape_1/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
model/reshape_1/Reshape_1/shapePack(model/reshape_1/strided_slice_1:output:0*model/reshape_1/Reshape_1/shape/1:output:0*model/reshape_1/Reshape_1/shape/2:output:0*model/reshape_1/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:¨
model/reshape_1/Reshape_1Reshapemodel/dense_1/BiasAdd:output:0(model/reshape_1/Reshape_1/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
-model/tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             É
(model/tf.compat.v1.transpose_1/transpose	Transpose model/reshape_1/Reshape:output:06model/tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
model/tf.compat.v1.shape/ShapeShape,model/tf.compat.v1.transpose_1/transpose:y:0*
T0*
_output_shapes
:
2model/tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ~
4model/tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ~
4model/tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
,model/tf.__operators__.getitem/strided_sliceStridedSlice'model/tf.compat.v1.shape/Shape:output:0;model/tf.__operators__.getitem/strided_slice/stack:output:0=model/tf.__operators__.getitem/strided_slice/stack_1:output:0=model/tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
model/tf.cast/CastCast5model/tf.__operators__.getitem/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
+model/tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ç
&model/tf.compat.v1.transpose/transpose	Transpose"model/reshape_1/Reshape_1:output:04model/tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
&model/dense_3/Tensordot/ReadVariableOpReadVariableOp/model_dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0f
model/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       z
model/dense_3/Tensordot/ShapeShape-model/layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
%model/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
 model/dense_3/Tensordot/GatherV2GatherV2&model/dense_3/Tensordot/Shape:output:0%model/dense_3/Tensordot/free:output:0.model/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
"model/dense_3/Tensordot/GatherV2_1GatherV2&model/dense_3/Tensordot/Shape:output:0%model/dense_3/Tensordot/axes:output:00model/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_3/Tensordot/ProdProd)model/dense_3/Tensordot/GatherV2:output:0&model/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_3/Tensordot/Prod_1Prod+model/dense_3/Tensordot/GatherV2_1:output:0(model/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
model/dense_3/Tensordot/concatConcatV2%model/dense_3/Tensordot/free:output:0%model/dense_3/Tensordot/axes:output:0,model/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:£
model/dense_3/Tensordot/stackPack%model/dense_3/Tensordot/Prod:output:0'model/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¼
!model/dense_3/Tensordot/transpose	Transpose-model/layer_normalization/batchnorm/add_1:z:0'model/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1´
model/dense_3/Tensordot/ReshapeReshape%model/dense_3/Tensordot/transpose:y:0&model/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
model/dense_3/Tensordot/MatMulMatMul(model/dense_3/Tensordot/Reshape:output:0.model/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
model/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
 model/dense_3/Tensordot/concat_1ConcatV2)model/dense_3/Tensordot/GatherV2:output:0(model/dense_3/Tensordot/Const_2:output:0.model/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:­
model/dense_3/TensordotReshape(model/dense_3/Tensordot/MatMul:product:0)model/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model/dense_3/BiasAddBiasAdd model/dense_3/Tensordot:output:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ï
model/tf.linalg.matmul/MatMulBatchMatMulV2*model/tf.compat.v1.transpose/transpose:y:0,model/tf.compat.v1.transpose_1/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(X
model/tf.math.sqrt/SqrtSqrtmodel/tf.cast/Cast:y:0*
T0*
_output_shapes
: e
model/reshape_1/Shape_2Shapemodel/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:o
%model/reshape_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model/reshape_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model/reshape_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
model/reshape_1/strided_slice_2StridedSlice model/reshape_1/Shape_2:output:0.model/reshape_1/strided_slice_2/stack:output:00model/reshape_1/strided_slice_2/stack_1:output:00model/reshape_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!model/reshape_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!model/reshape_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model/reshape_1/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
model/reshape_1/Reshape_2/shapePack(model/reshape_1/strided_slice_2:output:0*model/reshape_1/Reshape_2/shape/1:output:0*model/reshape_1/Reshape_2/shape/2:output:0*model/reshape_1/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:¨
model/reshape_1/Reshape_2Reshapemodel/dense_3/BiasAdd:output:0(model/reshape_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1§
model/tf.math.truediv/truedivRealDiv&model/tf.linalg.matmul/MatMul:output:0model/tf.math.sqrt/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
model/tf.nn.softmax/SoftmaxSoftmax!model/tf.math.truediv/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
-model/tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ë
(model/tf.compat.v1.transpose_2/transpose	Transpose"model/reshape_1/Reshape_2:output:06model/tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¿
model/tf.linalg.matmul_1/MatMulBatchMatMulV2%model/tf.nn.softmax/Softmax:softmax:0,model/tf.compat.v1.transpose_2/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
-model/tf.compat.v1.transpose_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ñ
(model/tf.compat.v1.transpose_3/transpose	Transpose(model/tf.linalg.matmul_1/MatMul:output:06model/tf.compat.v1.transpose_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1q
model/reshape_2/ShapeShape,model/tf.compat.v1.transpose_3/transpose:y:0*
T0*
_output_shapes
:m
#model/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
model/reshape_2/strided_sliceStridedSlicemodel/reshape_2/Shape:output:0,model/reshape_2/strided_slice/stack:output:0.model/reshape_2/strided_slice/stack_1:output:0.model/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
model/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
model/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Ï
model/reshape_2/Reshape/shapePack&model/reshape_2/strided_slice:output:0(model/reshape_2/Reshape/shape/1:output:0(model/reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:®
model/reshape_2/ReshapeReshape,model/tf.compat.v1.transpose_3/transpose:y:0&model/reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
&model/dense_4/Tensordot/ReadVariableOpReadVariableOp/model_dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0f
model/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       m
model/dense_4/Tensordot/ShapeShape model/reshape_2/Reshape:output:0*
T0*
_output_shapes
:g
%model/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
 model/dense_4/Tensordot/GatherV2GatherV2&model/dense_4/Tensordot/Shape:output:0%model/dense_4/Tensordot/free:output:0.model/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
"model/dense_4/Tensordot/GatherV2_1GatherV2&model/dense_4/Tensordot/Shape:output:0%model/dense_4/Tensordot/axes:output:00model/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_4/Tensordot/ProdProd)model/dense_4/Tensordot/GatherV2:output:0&model/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_4/Tensordot/Prod_1Prod+model/dense_4/Tensordot/GatherV2_1:output:0(model/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
model/dense_4/Tensordot/concatConcatV2%model/dense_4/Tensordot/free:output:0%model/dense_4/Tensordot/axes:output:0,model/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:£
model/dense_4/Tensordot/stackPack%model/dense_4/Tensordot/Prod:output:0'model/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¯
!model/dense_4/Tensordot/transpose	Transpose model/reshape_2/Reshape:output:0'model/dense_4/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1´
model/dense_4/Tensordot/ReshapeReshape%model/dense_4/Tensordot/transpose:y:0&model/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
model/dense_4/Tensordot/MatMulMatMul(model/dense_4/Tensordot/Reshape:output:0.model/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
model/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
 model/dense_4/Tensordot/concat_1ConcatV2)model/dense_4/Tensordot/GatherV2:output:0(model/dense_4/Tensordot/Const_2:output:0.model/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:­
model/dense_4/TensordotReshape(model/dense_4/Tensordot/MatMul:product:0)model/dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model/dense_4/BiasAddBiasAdd model/dense_4/Tensordot:output:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
model/dropout_1/IdentityIdentitymodel/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1³
"model/tf.__operators__.add_1/AddV2AddV2!model/dropout_1/Identity:output:0-model/layer_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
:model/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ä
(model/layer_normalization_1/moments/meanMean&model/tf.__operators__.add_1/AddV2:z:0Cmodel/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(©
0model/layer_normalization_1/moments/StopGradientStopGradient1model/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ã
5model/layer_normalization_1/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_1/AddV2:z:09model/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
>model/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ÿ
,model/layer_normalization_1/moments/varianceMean9model/layer_normalization_1/moments/SquaredDifference:z:0Gmodel/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(p
+model/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Õ
)model/layer_normalization_1/batchnorm/addAddV25model/layer_normalization_1/moments/variance:output:04model/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
+model/layer_normalization_1/batchnorm/RsqrtRsqrt-model/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¶
8model/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ù
)model/layer_normalization_1/batchnorm/mulMul/model/layer_normalization_1/batchnorm/Rsqrt:y:0@model/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¿
+model/layer_normalization_1/batchnorm/mul_1Mul&model/tf.__operators__.add_1/AddV2:z:0-model/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ê
+model/layer_normalization_1/batchnorm/mul_2Mul1model/layer_normalization_1/moments/mean:output:0-model/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1®
4model/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Õ
)model/layer_normalization_1/batchnorm/subSub<model/layer_normalization_1/batchnorm/ReadVariableOp:value:0/model/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ê
+model/layer_normalization_1/batchnorm/add_1AddV2/model/layer_normalization_1/batchnorm/mul_1:z:0-model/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
&model/dense_5/Tensordot/ReadVariableOpReadVariableOp/model_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0f
model/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
model/dense_5/Tensordot/ShapeShape/model/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
%model/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
 model/dense_5/Tensordot/GatherV2GatherV2&model/dense_5/Tensordot/Shape:output:0%model/dense_5/Tensordot/free:output:0.model/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
"model/dense_5/Tensordot/GatherV2_1GatherV2&model/dense_5/Tensordot/Shape:output:0%model/dense_5/Tensordot/axes:output:00model/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_5/Tensordot/ProdProd)model/dense_5/Tensordot/GatherV2:output:0&model/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_5/Tensordot/Prod_1Prod+model/dense_5/Tensordot/GatherV2_1:output:0(model/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
model/dense_5/Tensordot/concatConcatV2%model/dense_5/Tensordot/free:output:0%model/dense_5/Tensordot/axes:output:0,model/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:£
model/dense_5/Tensordot/stackPack%model/dense_5/Tensordot/Prod:output:0'model/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¾
!model/dense_5/Tensordot/transpose	Transpose/model/layer_normalization_1/batchnorm/add_1:z:0'model/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1´
model/dense_5/Tensordot/ReshapeReshape%model/dense_5/Tensordot/transpose:y:0&model/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
model/dense_5/Tensordot/MatMulMatMul(model/dense_5/Tensordot/Reshape:output:0.model/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
model/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
 model/dense_5/Tensordot/concat_1ConcatV2)model/dense_5/Tensordot/GatherV2:output:0(model/dense_5/Tensordot/Const_2:output:0.model/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:®
model/dense_5/TensordotReshape(model/dense_5/Tensordot/MatMul:product:0)model/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
model/dense_5/BiasAddBiasAdd model/dense_5/Tensordot:output:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
model/tf.nn.gelu/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
model/tf.nn.gelu/Gelu/mulMul$model/tf.nn.gelu/Gelu/mul/x:output:0model/dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
model/tf.nn.gelu/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?¦
model/tf.nn.gelu/Gelu/truedivRealDivmodel/dense_5/BiasAdd:output:0%model/tf.nn.gelu/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
model/tf.nn.gelu/Gelu/ErfErf!model/tf.nn.gelu/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
model/tf.nn.gelu/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/tf.nn.gelu/Gelu/addAddV2$model/tf.nn.gelu/Gelu/add/x:output:0model/tf.nn.gelu/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
model/tf.nn.gelu/Gelu/mul_1Mulmodel/tf.nn.gelu/Gelu/mul:z:0model/tf.nn.gelu/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
&model/dense_6/Tensordot/ReadVariableOpReadVariableOp/model_dense_6_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0f
model/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
model/dense_6/Tensordot/ShapeShapemodel/tf.nn.gelu/Gelu/mul_1:z:0*
T0*
_output_shapes
:g
%model/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
 model/dense_6/Tensordot/GatherV2GatherV2&model/dense_6/Tensordot/Shape:output:0%model/dense_6/Tensordot/free:output:0.model/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
"model/dense_6/Tensordot/GatherV2_1GatherV2&model/dense_6/Tensordot/Shape:output:0%model/dense_6/Tensordot/axes:output:00model/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_6/Tensordot/ProdProd)model/dense_6/Tensordot/GatherV2:output:0&model/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_6/Tensordot/Prod_1Prod+model/dense_6/Tensordot/GatherV2_1:output:0(model/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
model/dense_6/Tensordot/concatConcatV2%model/dense_6/Tensordot/free:output:0%model/dense_6/Tensordot/axes:output:0,model/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:£
model/dense_6/Tensordot/stackPack%model/dense_6/Tensordot/Prod:output:0'model/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¯
!model/dense_6/Tensordot/transpose	Transposemodel/tf.nn.gelu/Gelu/mul_1:z:0'model/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1´
model/dense_6/Tensordot/ReshapeReshape%model/dense_6/Tensordot/transpose:y:0&model/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
model/dense_6/Tensordot/MatMulMatMul(model/dense_6/Tensordot/Reshape:output:0.model/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
model/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
 model/dense_6/Tensordot/concat_1ConcatV2)model/dense_6/Tensordot/GatherV2:output:0(model/dense_6/Tensordot/Const_2:output:0.model/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:­
model/dense_6/TensordotReshape(model/dense_6/Tensordot/MatMul:product:0)model/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
$model/dense_6/BiasAdd/ReadVariableOpReadVariableOp-model_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model/dense_6/BiasAddBiasAdd model/dense_6/Tensordot:output:0,model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
model/dropout_2/IdentityIdentitymodel/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1µ
"model/tf.__operators__.add_2/AddV2AddV2!model/dropout_2/Identity:output:0/model/layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
:model/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ä
(model/layer_normalization_2/moments/meanMean&model/tf.__operators__.add_2/AddV2:z:0Cmodel/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(©
0model/layer_normalization_2/moments/StopGradientStopGradient1model/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ã
5model/layer_normalization_2/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_2/AddV2:z:09model/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
>model/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ÿ
,model/layer_normalization_2/moments/varianceMean9model/layer_normalization_2/moments/SquaredDifference:z:0Gmodel/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(p
+model/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Õ
)model/layer_normalization_2/batchnorm/addAddV25model/layer_normalization_2/moments/variance:output:04model/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
+model/layer_normalization_2/batchnorm/RsqrtRsqrt-model/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¶
8model/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ù
)model/layer_normalization_2/batchnorm/mulMul/model/layer_normalization_2/batchnorm/Rsqrt:y:0@model/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¿
+model/layer_normalization_2/batchnorm/mul_1Mul&model/tf.__operators__.add_2/AddV2:z:0-model/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ê
+model/layer_normalization_2/batchnorm/mul_2Mul1model/layer_normalization_2/moments/mean:output:0-model/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1®
4model/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Õ
)model/layer_normalization_2/batchnorm/subSub<model/layer_normalization_2/batchnorm/ReadVariableOp:value:0/model/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ê
+model/layer_normalization_2/batchnorm/add_1AddV2/model/layer_normalization_2/batchnorm/mul_1:z:0-model/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
&model/dense_8/Tensordot/ReadVariableOpReadVariableOp/model_dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0f
model/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
model/dense_8/Tensordot/ShapeShape/model/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
%model/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
 model/dense_8/Tensordot/GatherV2GatherV2&model/dense_8/Tensordot/Shape:output:0%model/dense_8/Tensordot/free:output:0.model/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
"model/dense_8/Tensordot/GatherV2_1GatherV2&model/dense_8/Tensordot/Shape:output:0%model/dense_8/Tensordot/axes:output:00model/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_8/Tensordot/ProdProd)model/dense_8/Tensordot/GatherV2:output:0&model/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_8/Tensordot/Prod_1Prod+model/dense_8/Tensordot/GatherV2_1:output:0(model/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
model/dense_8/Tensordot/concatConcatV2%model/dense_8/Tensordot/free:output:0%model/dense_8/Tensordot/axes:output:0,model/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:£
model/dense_8/Tensordot/stackPack%model/dense_8/Tensordot/Prod:output:0'model/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¾
!model/dense_8/Tensordot/transpose	Transpose/model/layer_normalization_2/batchnorm/add_1:z:0'model/dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1´
model/dense_8/Tensordot/ReshapeReshape%model/dense_8/Tensordot/transpose:y:0&model/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
model/dense_8/Tensordot/MatMulMatMul(model/dense_8/Tensordot/Reshape:output:0.model/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
model/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
 model/dense_8/Tensordot/concat_1ConcatV2)model/dense_8/Tensordot/GatherV2:output:0(model/dense_8/Tensordot/Const_2:output:0.model/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:­
model/dense_8/TensordotReshape(model/dense_8/Tensordot/MatMul:product:0)model/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
$model/dense_8/BiasAdd/ReadVariableOpReadVariableOp-model_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model/dense_8/BiasAddBiasAdd model/dense_8/Tensordot:output:0,model/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
&model/dense_7/Tensordot/ReadVariableOpReadVariableOp/model_dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0f
model/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
model/dense_7/Tensordot/ShapeShape/model/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
%model/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
 model/dense_7/Tensordot/GatherV2GatherV2&model/dense_7/Tensordot/Shape:output:0%model/dense_7/Tensordot/free:output:0.model/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
"model/dense_7/Tensordot/GatherV2_1GatherV2&model/dense_7/Tensordot/Shape:output:0%model/dense_7/Tensordot/axes:output:00model/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_7/Tensordot/ProdProd)model/dense_7/Tensordot/GatherV2:output:0&model/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_7/Tensordot/Prod_1Prod+model/dense_7/Tensordot/GatherV2_1:output:0(model/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
model/dense_7/Tensordot/concatConcatV2%model/dense_7/Tensordot/free:output:0%model/dense_7/Tensordot/axes:output:0,model/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:£
model/dense_7/Tensordot/stackPack%model/dense_7/Tensordot/Prod:output:0'model/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¾
!model/dense_7/Tensordot/transpose	Transpose/model/layer_normalization_2/batchnorm/add_1:z:0'model/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1´
model/dense_7/Tensordot/ReshapeReshape%model/dense_7/Tensordot/transpose:y:0&model/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
model/dense_7/Tensordot/MatMulMatMul(model/dense_7/Tensordot/Reshape:output:0.model/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
model/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
 model/dense_7/Tensordot/concat_1ConcatV2)model/dense_7/Tensordot/GatherV2:output:0(model/dense_7/Tensordot/Const_2:output:0.model/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:­
model/dense_7/TensordotReshape(model/dense_7/Tensordot/MatMul:product:0)model/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
$model/dense_7/BiasAdd/ReadVariableOpReadVariableOp-model_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model/dense_7/BiasAddBiasAdd model/dense_7/Tensordot:output:0,model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
model/reshape_3/ShapeShapemodel/dense_8/BiasAdd:output:0*
T0*
_output_shapes
:m
#model/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
model/reshape_3/strided_sliceStridedSlicemodel/reshape_3/Shape:output:0,model/reshape_3/strided_slice/stack:output:0.model/reshape_3/strided_slice/stack_1:output:0.model/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
model/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
model/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
model/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ù
model/reshape_3/Reshape/shapePack&model/reshape_3/strided_slice:output:0(model/reshape_3/Reshape/shape/1:output:0(model/reshape_3/Reshape/shape/2:output:0(model/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¤
model/reshape_3/ReshapeReshapemodel/dense_8/BiasAdd:output:0&model/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1e
model/reshape_3/Shape_1Shapemodel/dense_7/BiasAdd:output:0*
T0*
_output_shapes
:o
%model/reshape_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model/reshape_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model/reshape_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
model/reshape_3/strided_slice_1StridedSlice model/reshape_3/Shape_1:output:0.model/reshape_3/strided_slice_1/stack:output:00model/reshape_3/strided_slice_1/stack_1:output:00model/reshape_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!model/reshape_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!model/reshape_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model/reshape_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
model/reshape_3/Reshape_1/shapePack(model/reshape_3/strided_slice_1:output:0*model/reshape_3/Reshape_1/shape/1:output:0*model/reshape_3/Reshape_1/shape/2:output:0*model/reshape_3/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:¨
model/reshape_3/Reshape_1Reshapemodel/dense_7/BiasAdd:output:0(model/reshape_3/Reshape_1/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
-model/tf.compat.v1.transpose_5/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             É
(model/tf.compat.v1.transpose_5/transpose	Transpose model/reshape_3/Reshape:output:06model/tf.compat.v1.transpose_5/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1|
 model/tf.compat.v1.shape_1/ShapeShape,model/tf.compat.v1.transpose_5/transpose:y:0*
T0*
_output_shapes
:
4model/tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
6model/tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
6model/tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
.model/tf.__operators__.getitem_1/strided_sliceStridedSlice)model/tf.compat.v1.shape_1/Shape:output:0=model/tf.__operators__.getitem_1/strided_slice/stack:output:0?model/tf.__operators__.getitem_1/strided_slice/stack_1:output:0?model/tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
model/tf.cast_1/CastCast7model/tf.__operators__.getitem_1/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
-model/tf.compat.v1.transpose_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ë
(model/tf.compat.v1.transpose_4/transpose	Transpose"model/reshape_3/Reshape_1:output:06model/tf.compat.v1.transpose_4/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
&model/dense_9/Tensordot/ReadVariableOpReadVariableOp/model_dense_9_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0f
model/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       |
model/dense_9/Tensordot/ShapeShape/model/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:g
%model/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
 model/dense_9/Tensordot/GatherV2GatherV2&model/dense_9/Tensordot/Shape:output:0%model/dense_9/Tensordot/free:output:0.model/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
"model/dense_9/Tensordot/GatherV2_1GatherV2&model/dense_9/Tensordot/Shape:output:0%model/dense_9/Tensordot/axes:output:00model/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_9/Tensordot/ProdProd)model/dense_9/Tensordot/GatherV2:output:0&model/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_9/Tensordot/Prod_1Prod+model/dense_9/Tensordot/GatherV2_1:output:0(model/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
model/dense_9/Tensordot/concatConcatV2%model/dense_9/Tensordot/free:output:0%model/dense_9/Tensordot/axes:output:0,model/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:£
model/dense_9/Tensordot/stackPack%model/dense_9/Tensordot/Prod:output:0'model/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¾
!model/dense_9/Tensordot/transpose	Transpose/model/layer_normalization_2/batchnorm/add_1:z:0'model/dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1´
model/dense_9/Tensordot/ReshapeReshape%model/dense_9/Tensordot/transpose:y:0&model/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
model/dense_9/Tensordot/MatMulMatMul(model/dense_9/Tensordot/Reshape:output:0.model/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
model/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:g
%model/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
 model/dense_9/Tensordot/concat_1ConcatV2)model/dense_9/Tensordot/GatherV2:output:0(model/dense_9/Tensordot/Const_2:output:0.model/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:­
model/dense_9/TensordotReshape(model/dense_9/Tensordot/MatMul:product:0)model/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
$model/dense_9/BiasAdd/ReadVariableOpReadVariableOp-model_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¦
model/dense_9/BiasAddBiasAdd model/dense_9/Tensordot:output:0,model/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ó
model/tf.linalg.matmul_2/MatMulBatchMatMulV2,model/tf.compat.v1.transpose_4/transpose:y:0,model/tf.compat.v1.transpose_5/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(\
model/tf.math.sqrt_1/SqrtSqrtmodel/tf.cast_1/Cast:y:0*
T0*
_output_shapes
: e
model/reshape_3/Shape_2Shapemodel/dense_9/BiasAdd:output:0*
T0*
_output_shapes
:o
%model/reshape_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model/reshape_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model/reshape_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
model/reshape_3/strided_slice_2StridedSlice model/reshape_3/Shape_2:output:0.model/reshape_3/strided_slice_2/stack:output:00model/reshape_3/strided_slice_2/stack_1:output:00model/reshape_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!model/reshape_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!model/reshape_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model/reshape_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
model/reshape_3/Reshape_2/shapePack(model/reshape_3/strided_slice_2:output:0*model/reshape_3/Reshape_2/shape/1:output:0*model/reshape_3/Reshape_2/shape/2:output:0*model/reshape_3/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:¨
model/reshape_3/Reshape_2Reshapemodel/dense_9/BiasAdd:output:0(model/reshape_3/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
model/tf.math.truediv_1/truedivRealDiv(model/tf.linalg.matmul_2/MatMul:output:0model/tf.math.sqrt_1/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
model/tf.nn.softmax_1/SoftmaxSoftmax#model/tf.math.truediv_1/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
-model/tf.compat.v1.transpose_6/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ë
(model/tf.compat.v1.transpose_6/transpose	Transpose"model/reshape_3/Reshape_2:output:06model/tf.compat.v1.transpose_6/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Á
model/tf.linalg.matmul_3/MatMulBatchMatMulV2'model/tf.nn.softmax_1/Softmax:softmax:0,model/tf.compat.v1.transpose_6/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
-model/tf.compat.v1.transpose_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ñ
(model/tf.compat.v1.transpose_7/transpose	Transpose(model/tf.linalg.matmul_3/MatMul:output:06model/tf.compat.v1.transpose_7/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1q
model/reshape_4/ShapeShape,model/tf.compat.v1.transpose_7/transpose:y:0*
T0*
_output_shapes
:m
#model/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
model/reshape_4/strided_sliceStridedSlicemodel/reshape_4/Shape:output:0,model/reshape_4/strided_slice/stack:output:0.model/reshape_4/strided_slice/stack_1:output:0.model/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
model/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
model/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Ï
model/reshape_4/Reshape/shapePack&model/reshape_4/strided_slice:output:0(model/reshape_4/Reshape/shape/1:output:0(model/reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:®
model/reshape_4/ReshapeReshape,model/tf.compat.v1.transpose_7/transpose:y:0&model/reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'model/dense_10/Tensordot/ReadVariableOpReadVariableOp0model_dense_10_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0g
model/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
model/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
model/dense_10/Tensordot/ShapeShape model/reshape_4/Reshape:output:0*
T0*
_output_shapes
:h
&model/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!model/dense_10/Tensordot/GatherV2GatherV2'model/dense_10/Tensordot/Shape:output:0&model/dense_10/Tensordot/free:output:0/model/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(model/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#model/dense_10/Tensordot/GatherV2_1GatherV2'model/dense_10/Tensordot/Shape:output:0&model/dense_10/Tensordot/axes:output:01model/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
model/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_10/Tensordot/ProdProd*model/dense_10/Tensordot/GatherV2:output:0'model/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 model/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
model/dense_10/Tensordot/Prod_1Prod,model/dense_10/Tensordot/GatherV2_1:output:0)model/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$model/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
model/dense_10/Tensordot/concatConcatV2&model/dense_10/Tensordot/free:output:0&model/dense_10/Tensordot/axes:output:0-model/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
model/dense_10/Tensordot/stackPack&model/dense_10/Tensordot/Prod:output:0(model/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:±
"model/dense_10/Tensordot/transpose	Transpose model/reshape_4/Reshape:output:0(model/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1·
 model/dense_10/Tensordot/ReshapeReshape&model/dense_10/Tensordot/transpose:y:0'model/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
model/dense_10/Tensordot/MatMulMatMul)model/dense_10/Tensordot/Reshape:output:0/model/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
 model/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&model/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!model/dense_10/Tensordot/concat_1ConcatV2*model/dense_10/Tensordot/GatherV2:output:0)model/dense_10/Tensordot/Const_2:output:0/model/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:°
model/dense_10/TensordotReshape)model/dense_10/Tensordot/MatMul:product:0*model/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%model/dense_10/BiasAdd/ReadVariableOpReadVariableOp.model_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model/dense_10/BiasAddBiasAdd!model/dense_10/Tensordot:output:0-model/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1{
model/dropout_3/IdentityIdentitymodel/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1µ
"model/tf.__operators__.add_3/AddV2AddV2!model/dropout_3/Identity:output:0/model/layer_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
:model/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ä
(model/layer_normalization_3/moments/meanMean&model/tf.__operators__.add_3/AddV2:z:0Cmodel/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(©
0model/layer_normalization_3/moments/StopGradientStopGradient1model/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ã
5model/layer_normalization_3/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_3/AddV2:z:09model/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
>model/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ÿ
,model/layer_normalization_3/moments/varianceMean9model/layer_normalization_3/moments/SquaredDifference:z:0Gmodel/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(p
+model/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Õ
)model/layer_normalization_3/batchnorm/addAddV25model/layer_normalization_3/moments/variance:output:04model/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
+model/layer_normalization_3/batchnorm/RsqrtRsqrt-model/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¶
8model/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ù
)model/layer_normalization_3/batchnorm/mulMul/model/layer_normalization_3/batchnorm/Rsqrt:y:0@model/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¿
+model/layer_normalization_3/batchnorm/mul_1Mul&model/tf.__operators__.add_3/AddV2:z:0-model/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ê
+model/layer_normalization_3/batchnorm/mul_2Mul1model/layer_normalization_3/moments/mean:output:0-model/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1®
4model/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Õ
)model/layer_normalization_3/batchnorm/subSub<model/layer_normalization_3/batchnorm/ReadVariableOp:value:0/model/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ê
+model/layer_normalization_3/batchnorm/add_1AddV2/model/layer_normalization_3/batchnorm/mul_1:z:0-model/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'model/dense_11/Tensordot/ReadVariableOpReadVariableOp0model_dense_11_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0g
model/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
model/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
model/dense_11/Tensordot/ShapeShape/model/layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:h
&model/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!model/dense_11/Tensordot/GatherV2GatherV2'model/dense_11/Tensordot/Shape:output:0&model/dense_11/Tensordot/free:output:0/model/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(model/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#model/dense_11/Tensordot/GatherV2_1GatherV2'model/dense_11/Tensordot/Shape:output:0&model/dense_11/Tensordot/axes:output:01model/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
model/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_11/Tensordot/ProdProd*model/dense_11/Tensordot/GatherV2:output:0'model/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 model/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
model/dense_11/Tensordot/Prod_1Prod,model/dense_11/Tensordot/GatherV2_1:output:0)model/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$model/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
model/dense_11/Tensordot/concatConcatV2&model/dense_11/Tensordot/free:output:0&model/dense_11/Tensordot/axes:output:0-model/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
model/dense_11/Tensordot/stackPack&model/dense_11/Tensordot/Prod:output:0(model/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:À
"model/dense_11/Tensordot/transpose	Transpose/model/layer_normalization_3/batchnorm/add_1:z:0(model/dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1·
 model/dense_11/Tensordot/ReshapeReshape&model/dense_11/Tensordot/transpose:y:0'model/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
model/dense_11/Tensordot/MatMulMatMul)model/dense_11/Tensordot/Reshape:output:0/model/dense_11/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 model/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&model/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!model/dense_11/Tensordot/concat_1ConcatV2*model/dense_11/Tensordot/GatherV2:output:0)model/dense_11/Tensordot/Const_2:output:0/model/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:±
model/dense_11/TensordotReshape)model/dense_11/Tensordot/MatMul:product:0*model/dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%model/dense_11/BiasAdd/ReadVariableOpReadVariableOp.model_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/dense_11/BiasAddBiasAdd!model/dense_11/Tensordot:output:0-model/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1b
model/tf.nn.gelu_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¢
model/tf.nn.gelu_1/Gelu/mulMul&model/tf.nn.gelu_1/Gelu/mul/x:output:0model/dense_11/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
model/tf.nn.gelu_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?«
model/tf.nn.gelu_1/Gelu/truedivRealDivmodel/dense_11/BiasAdd:output:0'model/tf.nn.gelu_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
model/tf.nn.gelu_1/Gelu/ErfErf#model/tf.nn.gelu_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1b
model/tf.nn.gelu_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
model/tf.nn.gelu_1/Gelu/addAddV2&model/tf.nn.gelu_1/Gelu/add/x:output:0model/tf.nn.gelu_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
model/tf.nn.gelu_1/Gelu/mul_1Mulmodel/tf.nn.gelu_1/Gelu/mul:z:0model/tf.nn.gelu_1/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'model/dense_12/Tensordot/ReadVariableOpReadVariableOp0model_dense_12_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0g
model/dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
model/dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       o
model/dense_12/Tensordot/ShapeShape!model/tf.nn.gelu_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:h
&model/dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!model/dense_12/Tensordot/GatherV2GatherV2'model/dense_12/Tensordot/Shape:output:0&model/dense_12/Tensordot/free:output:0/model/dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(model/dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#model/dense_12/Tensordot/GatherV2_1GatherV2'model/dense_12/Tensordot/Shape:output:0&model/dense_12/Tensordot/axes:output:01model/dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
model/dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_12/Tensordot/ProdProd*model/dense_12/Tensordot/GatherV2:output:0'model/dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 model/dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
model/dense_12/Tensordot/Prod_1Prod,model/dense_12/Tensordot/GatherV2_1:output:0)model/dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$model/dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
model/dense_12/Tensordot/concatConcatV2&model/dense_12/Tensordot/free:output:0&model/dense_12/Tensordot/axes:output:0-model/dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
model/dense_12/Tensordot/stackPack&model/dense_12/Tensordot/Prod:output:0(model/dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:³
"model/dense_12/Tensordot/transpose	Transpose!model/tf.nn.gelu_1/Gelu/mul_1:z:0(model/dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1·
 model/dense_12/Tensordot/ReshapeReshape&model/dense_12/Tensordot/transpose:y:0'model/dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
model/dense_12/Tensordot/MatMulMatMul)model/dense_12/Tensordot/Reshape:output:0/model/dense_12/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
 model/dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&model/dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!model/dense_12/Tensordot/concat_1ConcatV2*model/dense_12/Tensordot/GatherV2:output:0)model/dense_12/Tensordot/Const_2:output:0/model/dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:°
model/dense_12/TensordotReshape)model/dense_12/Tensordot/MatMul:product:0*model/dense_12/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%model/dense_12/BiasAdd/ReadVariableOpReadVariableOp.model_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model/dense_12/BiasAddBiasAdd!model/dense_12/Tensordot:output:0-model/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1{
model/dropout_4/IdentityIdentitymodel/dense_12/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1µ
"model/tf.__operators__.add_4/AddV2AddV2!model/dropout_4/Identity:output:0/model/layer_normalization_3/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
:model/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ä
(model/layer_normalization_4/moments/meanMean&model/tf.__operators__.add_4/AddV2:z:0Cmodel/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(©
0model/layer_normalization_4/moments/StopGradientStopGradient1model/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ã
5model/layer_normalization_4/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_4/AddV2:z:09model/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
>model/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ÿ
,model/layer_normalization_4/moments/varianceMean9model/layer_normalization_4/moments/SquaredDifference:z:0Gmodel/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(p
+model/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Õ
)model/layer_normalization_4/batchnorm/addAddV25model/layer_normalization_4/moments/variance:output:04model/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
+model/layer_normalization_4/batchnorm/RsqrtRsqrt-model/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¶
8model/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ù
)model/layer_normalization_4/batchnorm/mulMul/model/layer_normalization_4/batchnorm/Rsqrt:y:0@model/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¿
+model/layer_normalization_4/batchnorm/mul_1Mul&model/tf.__operators__.add_4/AddV2:z:0-model/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ê
+model/layer_normalization_4/batchnorm/mul_2Mul1model/layer_normalization_4/moments/mean:output:0-model/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1®
4model/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Õ
)model/layer_normalization_4/batchnorm/subSub<model/layer_normalization_4/batchnorm/ReadVariableOp:value:0/model/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ê
+model/layer_normalization_4/batchnorm/add_1AddV2/model/layer_normalization_4/batchnorm/mul_1:z:0-model/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'model/dense_14/Tensordot/ReadVariableOpReadVariableOp0model_dense_14_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0g
model/dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
model/dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
model/dense_14/Tensordot/ShapeShape/model/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:h
&model/dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!model/dense_14/Tensordot/GatherV2GatherV2'model/dense_14/Tensordot/Shape:output:0&model/dense_14/Tensordot/free:output:0/model/dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(model/dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#model/dense_14/Tensordot/GatherV2_1GatherV2'model/dense_14/Tensordot/Shape:output:0&model/dense_14/Tensordot/axes:output:01model/dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
model/dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_14/Tensordot/ProdProd*model/dense_14/Tensordot/GatherV2:output:0'model/dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 model/dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
model/dense_14/Tensordot/Prod_1Prod,model/dense_14/Tensordot/GatherV2_1:output:0)model/dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$model/dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
model/dense_14/Tensordot/concatConcatV2&model/dense_14/Tensordot/free:output:0&model/dense_14/Tensordot/axes:output:0-model/dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
model/dense_14/Tensordot/stackPack&model/dense_14/Tensordot/Prod:output:0(model/dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:À
"model/dense_14/Tensordot/transpose	Transpose/model/layer_normalization_4/batchnorm/add_1:z:0(model/dense_14/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1·
 model/dense_14/Tensordot/ReshapeReshape&model/dense_14/Tensordot/transpose:y:0'model/dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
model/dense_14/Tensordot/MatMulMatMul)model/dense_14/Tensordot/Reshape:output:0/model/dense_14/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
 model/dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&model/dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!model/dense_14/Tensordot/concat_1ConcatV2*model/dense_14/Tensordot/GatherV2:output:0)model/dense_14/Tensordot/Const_2:output:0/model/dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:°
model/dense_14/TensordotReshape)model/dense_14/Tensordot/MatMul:product:0*model/dense_14/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%model/dense_14/BiasAdd/ReadVariableOpReadVariableOp.model_dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model/dense_14/BiasAddBiasAdd!model/dense_14/Tensordot:output:0-model/dense_14/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'model/dense_13/Tensordot/ReadVariableOpReadVariableOp0model_dense_13_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0g
model/dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
model/dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
model/dense_13/Tensordot/ShapeShape/model/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:h
&model/dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!model/dense_13/Tensordot/GatherV2GatherV2'model/dense_13/Tensordot/Shape:output:0&model/dense_13/Tensordot/free:output:0/model/dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(model/dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#model/dense_13/Tensordot/GatherV2_1GatherV2'model/dense_13/Tensordot/Shape:output:0&model/dense_13/Tensordot/axes:output:01model/dense_13/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
model/dense_13/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_13/Tensordot/ProdProd*model/dense_13/Tensordot/GatherV2:output:0'model/dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 model/dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
model/dense_13/Tensordot/Prod_1Prod,model/dense_13/Tensordot/GatherV2_1:output:0)model/dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$model/dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
model/dense_13/Tensordot/concatConcatV2&model/dense_13/Tensordot/free:output:0&model/dense_13/Tensordot/axes:output:0-model/dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
model/dense_13/Tensordot/stackPack&model/dense_13/Tensordot/Prod:output:0(model/dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:À
"model/dense_13/Tensordot/transpose	Transpose/model/layer_normalization_4/batchnorm/add_1:z:0(model/dense_13/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1·
 model/dense_13/Tensordot/ReshapeReshape&model/dense_13/Tensordot/transpose:y:0'model/dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
model/dense_13/Tensordot/MatMulMatMul)model/dense_13/Tensordot/Reshape:output:0/model/dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
 model/dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&model/dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!model/dense_13/Tensordot/concat_1ConcatV2*model/dense_13/Tensordot/GatherV2:output:0)model/dense_13/Tensordot/Const_2:output:0/model/dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:°
model/dense_13/TensordotReshape)model/dense_13/Tensordot/MatMul:product:0*model/dense_13/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%model/dense_13/BiasAdd/ReadVariableOpReadVariableOp.model_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model/dense_13/BiasAddBiasAdd!model/dense_13/Tensordot:output:0-model/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1d
model/reshape_5/ShapeShapemodel/dense_14/BiasAdd:output:0*
T0*
_output_shapes
:m
#model/reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
model/reshape_5/strided_sliceStridedSlicemodel/reshape_5/Shape:output:0,model/reshape_5/strided_slice/stack:output:0.model/reshape_5/strided_slice/stack_1:output:0.model/reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
model/reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
model/reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
model/reshape_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :ù
model/reshape_5/Reshape/shapePack&model/reshape_5/strided_slice:output:0(model/reshape_5/Reshape/shape/1:output:0(model/reshape_5/Reshape/shape/2:output:0(model/reshape_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¥
model/reshape_5/ReshapeReshapemodel/dense_14/BiasAdd:output:0&model/reshape_5/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
model/reshape_5/Shape_1Shapemodel/dense_13/BiasAdd:output:0*
T0*
_output_shapes
:o
%model/reshape_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model/reshape_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model/reshape_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
model/reshape_5/strided_slice_1StridedSlice model/reshape_5/Shape_1:output:0.model/reshape_5/strided_slice_1/stack:output:00model/reshape_5/strided_slice_1/stack_1:output:00model/reshape_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!model/reshape_5/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!model/reshape_5/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model/reshape_5/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
model/reshape_5/Reshape_1/shapePack(model/reshape_5/strided_slice_1:output:0*model/reshape_5/Reshape_1/shape/1:output:0*model/reshape_5/Reshape_1/shape/2:output:0*model/reshape_5/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:©
model/reshape_5/Reshape_1Reshapemodel/dense_13/BiasAdd:output:0(model/reshape_5/Reshape_1/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
-model/tf.compat.v1.transpose_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             É
(model/tf.compat.v1.transpose_9/transpose	Transpose model/reshape_5/Reshape:output:06model/tf.compat.v1.transpose_9/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1|
 model/tf.compat.v1.shape_2/ShapeShape,model/tf.compat.v1.transpose_9/transpose:y:0*
T0*
_output_shapes
:
4model/tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
6model/tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
6model/tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ð
.model/tf.__operators__.getitem_2/strided_sliceStridedSlice)model/tf.compat.v1.shape_2/Shape:output:0=model/tf.__operators__.getitem_2/strided_slice/stack:output:0?model/tf.__operators__.getitem_2/strided_slice/stack_1:output:0?model/tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
model/tf.cast_2/CastCast7model/tf.__operators__.getitem_2/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
-model/tf.compat.v1.transpose_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ë
(model/tf.compat.v1.transpose_8/transpose	Transpose"model/reshape_5/Reshape_1:output:06model/tf.compat.v1.transpose_8/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'model/dense_15/Tensordot/ReadVariableOpReadVariableOp0model_dense_15_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0g
model/dense_15/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
model/dense_15/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
model/dense_15/Tensordot/ShapeShape/model/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:h
&model/dense_15/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!model/dense_15/Tensordot/GatherV2GatherV2'model/dense_15/Tensordot/Shape:output:0&model/dense_15/Tensordot/free:output:0/model/dense_15/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(model/dense_15/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#model/dense_15/Tensordot/GatherV2_1GatherV2'model/dense_15/Tensordot/Shape:output:0&model/dense_15/Tensordot/axes:output:01model/dense_15/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
model/dense_15/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_15/Tensordot/ProdProd*model/dense_15/Tensordot/GatherV2:output:0'model/dense_15/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 model/dense_15/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
model/dense_15/Tensordot/Prod_1Prod,model/dense_15/Tensordot/GatherV2_1:output:0)model/dense_15/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$model/dense_15/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
model/dense_15/Tensordot/concatConcatV2&model/dense_15/Tensordot/free:output:0&model/dense_15/Tensordot/axes:output:0-model/dense_15/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
model/dense_15/Tensordot/stackPack&model/dense_15/Tensordot/Prod:output:0(model/dense_15/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:À
"model/dense_15/Tensordot/transpose	Transpose/model/layer_normalization_4/batchnorm/add_1:z:0(model/dense_15/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1·
 model/dense_15/Tensordot/ReshapeReshape&model/dense_15/Tensordot/transpose:y:0'model/dense_15/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
model/dense_15/Tensordot/MatMulMatMul)model/dense_15/Tensordot/Reshape:output:0/model/dense_15/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
 model/dense_15/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&model/dense_15/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!model/dense_15/Tensordot/concat_1ConcatV2*model/dense_15/Tensordot/GatherV2:output:0)model/dense_15/Tensordot/Const_2:output:0/model/dense_15/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:°
model/dense_15/TensordotReshape)model/dense_15/Tensordot/MatMul:product:0*model/dense_15/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%model/dense_15/BiasAdd/ReadVariableOpReadVariableOp.model_dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model/dense_15/BiasAddBiasAdd!model/dense_15/Tensordot:output:0-model/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ó
model/tf.linalg.matmul_4/MatMulBatchMatMulV2,model/tf.compat.v1.transpose_8/transpose:y:0,model/tf.compat.v1.transpose_9/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(\
model/tf.math.sqrt_2/SqrtSqrtmodel/tf.cast_2/Cast:y:0*
T0*
_output_shapes
: f
model/reshape_5/Shape_2Shapemodel/dense_15/BiasAdd:output:0*
T0*
_output_shapes
:o
%model/reshape_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model/reshape_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model/reshape_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
model/reshape_5/strided_slice_2StridedSlice model/reshape_5/Shape_2:output:0.model/reshape_5/strided_slice_2/stack:output:00model/reshape_5/strided_slice_2/stack_1:output:00model/reshape_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!model/reshape_5/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿc
!model/reshape_5/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :c
!model/reshape_5/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
model/reshape_5/Reshape_2/shapePack(model/reshape_5/strided_slice_2:output:0*model/reshape_5/Reshape_2/shape/1:output:0*model/reshape_5/Reshape_2/shape/2:output:0*model/reshape_5/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:©
model/reshape_5/Reshape_2Reshapemodel/dense_15/BiasAdd:output:0(model/reshape_5/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
model/tf.math.truediv_2/truedivRealDiv(model/tf.linalg.matmul_4/MatMul:output:0model/tf.math.sqrt_2/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
model/tf.nn.softmax_2/SoftmaxSoftmax#model/tf.math.truediv_2/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
.model/tf.compat.v1.transpose_10/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Í
)model/tf.compat.v1.transpose_10/transpose	Transpose"model/reshape_5/Reshape_2:output:07model/tf.compat.v1.transpose_10/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
model/tf.linalg.matmul_5/MatMulBatchMatMulV2'model/tf.nn.softmax_2/Softmax:softmax:0-model/tf.compat.v1.transpose_10/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
.model/tf.compat.v1.transpose_11/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ó
)model/tf.compat.v1.transpose_11/transpose	Transpose(model/tf.linalg.matmul_5/MatMul:output:07model/tf.compat.v1.transpose_11/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
model/reshape_6/ShapeShape-model/tf.compat.v1.transpose_11/transpose:y:0*
T0*
_output_shapes
:m
#model/reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model/reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model/reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
model/reshape_6/strided_sliceStridedSlicemodel/reshape_6/Shape:output:0,model/reshape_6/strided_slice/stack:output:0.model/reshape_6/strided_slice/stack_1:output:0.model/reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
model/reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿa
model/reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Ï
model/reshape_6/Reshape/shapePack&model/reshape_6/strided_slice:output:0(model/reshape_6/Reshape/shape/1:output:0(model/reshape_6/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:¯
model/reshape_6/ReshapeReshape-model/tf.compat.v1.transpose_11/transpose:y:0&model/reshape_6/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'model/dense_16/Tensordot/ReadVariableOpReadVariableOp0model_dense_16_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0g
model/dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
model/dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
model/dense_16/Tensordot/ShapeShape model/reshape_6/Reshape:output:0*
T0*
_output_shapes
:h
&model/dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!model/dense_16/Tensordot/GatherV2GatherV2'model/dense_16/Tensordot/Shape:output:0&model/dense_16/Tensordot/free:output:0/model/dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(model/dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#model/dense_16/Tensordot/GatherV2_1GatherV2'model/dense_16/Tensordot/Shape:output:0&model/dense_16/Tensordot/axes:output:01model/dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
model/dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_16/Tensordot/ProdProd*model/dense_16/Tensordot/GatherV2:output:0'model/dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 model/dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
model/dense_16/Tensordot/Prod_1Prod,model/dense_16/Tensordot/GatherV2_1:output:0)model/dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$model/dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
model/dense_16/Tensordot/concatConcatV2&model/dense_16/Tensordot/free:output:0&model/dense_16/Tensordot/axes:output:0-model/dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
model/dense_16/Tensordot/stackPack&model/dense_16/Tensordot/Prod:output:0(model/dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:±
"model/dense_16/Tensordot/transpose	Transpose model/reshape_6/Reshape:output:0(model/dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1·
 model/dense_16/Tensordot/ReshapeReshape&model/dense_16/Tensordot/transpose:y:0'model/dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
model/dense_16/Tensordot/MatMulMatMul)model/dense_16/Tensordot/Reshape:output:0/model/dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
 model/dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&model/dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!model/dense_16/Tensordot/concat_1ConcatV2*model/dense_16/Tensordot/GatherV2:output:0)model/dense_16/Tensordot/Const_2:output:0/model/dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:°
model/dense_16/TensordotReshape)model/dense_16/Tensordot/MatMul:product:0*model/dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%model/dense_16/BiasAdd/ReadVariableOpReadVariableOp.model_dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model/dense_16/BiasAddBiasAdd!model/dense_16/Tensordot:output:0-model/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1{
model/dropout_5/IdentityIdentitymodel/dense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1µ
"model/tf.__operators__.add_5/AddV2AddV2!model/dropout_5/Identity:output:0/model/layer_normalization_4/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
:model/layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ä
(model/layer_normalization_5/moments/meanMean&model/tf.__operators__.add_5/AddV2:z:0Cmodel/layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(©
0model/layer_normalization_5/moments/StopGradientStopGradient1model/layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ã
5model/layer_normalization_5/moments/SquaredDifferenceSquaredDifference&model/tf.__operators__.add_5/AddV2:z:09model/layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
>model/layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ÿ
,model/layer_normalization_5/moments/varianceMean9model/layer_normalization_5/moments/SquaredDifference:z:0Gmodel/layer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(p
+model/layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Õ
)model/layer_normalization_5/batchnorm/addAddV25model/layer_normalization_5/moments/variance:output:04model/layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
+model/layer_normalization_5/batchnorm/RsqrtRsqrt-model/layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¶
8model/layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ù
)model/layer_normalization_5/batchnorm/mulMul/model/layer_normalization_5/batchnorm/Rsqrt:y:0@model/layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¿
+model/layer_normalization_5/batchnorm/mul_1Mul&model/tf.__operators__.add_5/AddV2:z:0-model/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ê
+model/layer_normalization_5/batchnorm/mul_2Mul1model/layer_normalization_5/moments/mean:output:0-model/layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1®
4model/layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp=model_layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Õ
)model/layer_normalization_5/batchnorm/subSub<model/layer_normalization_5/batchnorm/ReadVariableOp:value:0/model/layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ê
+model/layer_normalization_5/batchnorm/add_1AddV2/model/layer_normalization_5/batchnorm/mul_1:z:0-model/layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'model/dense_17/Tensordot/ReadVariableOpReadVariableOp0model_dense_17_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0g
model/dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
model/dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
model/dense_17/Tensordot/ShapeShape/model/layer_normalization_5/batchnorm/add_1:z:0*
T0*
_output_shapes
:h
&model/dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!model/dense_17/Tensordot/GatherV2GatherV2'model/dense_17/Tensordot/Shape:output:0&model/dense_17/Tensordot/free:output:0/model/dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(model/dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#model/dense_17/Tensordot/GatherV2_1GatherV2'model/dense_17/Tensordot/Shape:output:0&model/dense_17/Tensordot/axes:output:01model/dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
model/dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_17/Tensordot/ProdProd*model/dense_17/Tensordot/GatherV2:output:0'model/dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 model/dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
model/dense_17/Tensordot/Prod_1Prod,model/dense_17/Tensordot/GatherV2_1:output:0)model/dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$model/dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
model/dense_17/Tensordot/concatConcatV2&model/dense_17/Tensordot/free:output:0&model/dense_17/Tensordot/axes:output:0-model/dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
model/dense_17/Tensordot/stackPack&model/dense_17/Tensordot/Prod:output:0(model/dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:À
"model/dense_17/Tensordot/transpose	Transpose/model/layer_normalization_5/batchnorm/add_1:z:0(model/dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1·
 model/dense_17/Tensordot/ReshapeReshape&model/dense_17/Tensordot/transpose:y:0'model/dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¸
model/dense_17/Tensordot/MatMulMatMul)model/dense_17/Tensordot/Reshape:output:0/model/dense_17/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
 model/dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&model/dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!model/dense_17/Tensordot/concat_1ConcatV2*model/dense_17/Tensordot/GatherV2:output:0)model/dense_17/Tensordot/Const_2:output:0/model/dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:±
model/dense_17/TensordotReshape)model/dense_17/Tensordot/MatMul:product:0*model/dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%model/dense_17/BiasAdd/ReadVariableOpReadVariableOp.model_dense_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/dense_17/BiasAddBiasAdd!model/dense_17/Tensordot:output:0-model/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1b
model/tf.nn.gelu_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¢
model/tf.nn.gelu_2/Gelu/mulMul&model/tf.nn.gelu_2/Gelu/mul/x:output:0model/dense_17/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
model/tf.nn.gelu_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?«
model/tf.nn.gelu_2/Gelu/truedivRealDivmodel/dense_17/BiasAdd:output:0'model/tf.nn.gelu_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
model/tf.nn.gelu_2/Gelu/ErfErf#model/tf.nn.gelu_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1b
model/tf.nn.gelu_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
model/tf.nn.gelu_2/Gelu/addAddV2&model/tf.nn.gelu_2/Gelu/add/x:output:0model/tf.nn.gelu_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
model/tf.nn.gelu_2/Gelu/mul_1Mulmodel/tf.nn.gelu_2/Gelu/mul:z:0model/tf.nn.gelu_2/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'model/dense_18/Tensordot/ReadVariableOpReadVariableOp0model_dense_18_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0g
model/dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:n
model/dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       o
model/dense_18/Tensordot/ShapeShape!model/tf.nn.gelu_2/Gelu/mul_1:z:0*
T0*
_output_shapes
:h
&model/dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
!model/dense_18/Tensordot/GatherV2GatherV2'model/dense_18/Tensordot/Shape:output:0&model/dense_18/Tensordot/free:output:0/model/dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
(model/dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
#model/dense_18/Tensordot/GatherV2_1GatherV2'model/dense_18/Tensordot/Shape:output:0&model/dense_18/Tensordot/axes:output:01model/dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
model/dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_18/Tensordot/ProdProd*model/dense_18/Tensordot/GatherV2:output:0'model/dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: j
 model/dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¡
model/dense_18/Tensordot/Prod_1Prod,model/dense_18/Tensordot/GatherV2_1:output:0)model/dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: f
$model/dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
model/dense_18/Tensordot/concatConcatV2&model/dense_18/Tensordot/free:output:0&model/dense_18/Tensordot/axes:output:0-model/dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¦
model/dense_18/Tensordot/stackPack&model/dense_18/Tensordot/Prod:output:0(model/dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:³
"model/dense_18/Tensordot/transpose	Transpose!model/tf.nn.gelu_2/Gelu/mul_1:z:0(model/dense_18/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1·
 model/dense_18/Tensordot/ReshapeReshape&model/dense_18/Tensordot/transpose:y:0'model/dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
model/dense_18/Tensordot/MatMulMatMul)model/dense_18/Tensordot/Reshape:output:0/model/dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
 model/dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:h
&model/dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
!model/dense_18/Tensordot/concat_1ConcatV2*model/dense_18/Tensordot/GatherV2:output:0)model/dense_18/Tensordot/Const_2:output:0/model/dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:°
model/dense_18/TensordotReshape)model/dense_18/Tensordot/MatMul:product:0*model/dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%model/dense_18/BiasAdd/ReadVariableOpReadVariableOp.model_dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
model/dense_18/BiasAddBiasAdd!model/dense_18/Tensordot:output:0-model/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1{
model/dropout_6/IdentityIdentitymodel/dense_18/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1µ
"model/tf.__operators__.add_6/AddV2AddV2!model/dropout_6/Identity:output:0/model/layer_normalization_5/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
model/flatten/ReshapeReshape&model/tf.__operators__.add_6/AddV2:z:0model/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
model/softmax/SoftmaxSoftmaxmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
IdentityIdentitymodel/softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
é
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp'^model/dense_1/Tensordot/ReadVariableOp&^model/dense_10/BiasAdd/ReadVariableOp(^model/dense_10/Tensordot/ReadVariableOp&^model/dense_11/BiasAdd/ReadVariableOp(^model/dense_11/Tensordot/ReadVariableOp&^model/dense_12/BiasAdd/ReadVariableOp(^model/dense_12/Tensordot/ReadVariableOp&^model/dense_13/BiasAdd/ReadVariableOp(^model/dense_13/Tensordot/ReadVariableOp&^model/dense_14/BiasAdd/ReadVariableOp(^model/dense_14/Tensordot/ReadVariableOp&^model/dense_15/BiasAdd/ReadVariableOp(^model/dense_15/Tensordot/ReadVariableOp&^model/dense_16/BiasAdd/ReadVariableOp(^model/dense_16/Tensordot/ReadVariableOp&^model/dense_17/BiasAdd/ReadVariableOp(^model/dense_17/Tensordot/ReadVariableOp&^model/dense_18/BiasAdd/ReadVariableOp(^model/dense_18/Tensordot/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp'^model/dense_2/Tensordot/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp'^model/dense_3/Tensordot/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp'^model/dense_4/Tensordot/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp'^model/dense_5/Tensordot/ReadVariableOp%^model/dense_6/BiasAdd/ReadVariableOp'^model/dense_6/Tensordot/ReadVariableOp%^model/dense_7/BiasAdd/ReadVariableOp'^model/dense_7/Tensordot/ReadVariableOp%^model/dense_8/BiasAdd/ReadVariableOp'^model/dense_8/Tensordot/ReadVariableOp%^model/dense_9/BiasAdd/ReadVariableOp'^model/dense_9/Tensordot/ReadVariableOp3^model/layer_normalization/batchnorm/ReadVariableOp7^model/layer_normalization/batchnorm/mul/ReadVariableOp5^model/layer_normalization_1/batchnorm/ReadVariableOp9^model/layer_normalization_1/batchnorm/mul/ReadVariableOp5^model/layer_normalization_2/batchnorm/ReadVariableOp9^model/layer_normalization_2/batchnorm/mul/ReadVariableOp5^model/layer_normalization_3/batchnorm/ReadVariableOp9^model/layer_normalization_3/batchnorm/mul/ReadVariableOp5^model/layer_normalization_4/batchnorm/ReadVariableOp9^model/layer_normalization_4/batchnorm/mul/ReadVariableOp5^model/layer_normalization_5/batchnorm/ReadVariableOp9^model/layer_normalization_5/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2P
&model/dense_1/Tensordot/ReadVariableOp&model/dense_1/Tensordot/ReadVariableOp2N
%model/dense_10/BiasAdd/ReadVariableOp%model/dense_10/BiasAdd/ReadVariableOp2R
'model/dense_10/Tensordot/ReadVariableOp'model/dense_10/Tensordot/ReadVariableOp2N
%model/dense_11/BiasAdd/ReadVariableOp%model/dense_11/BiasAdd/ReadVariableOp2R
'model/dense_11/Tensordot/ReadVariableOp'model/dense_11/Tensordot/ReadVariableOp2N
%model/dense_12/BiasAdd/ReadVariableOp%model/dense_12/BiasAdd/ReadVariableOp2R
'model/dense_12/Tensordot/ReadVariableOp'model/dense_12/Tensordot/ReadVariableOp2N
%model/dense_13/BiasAdd/ReadVariableOp%model/dense_13/BiasAdd/ReadVariableOp2R
'model/dense_13/Tensordot/ReadVariableOp'model/dense_13/Tensordot/ReadVariableOp2N
%model/dense_14/BiasAdd/ReadVariableOp%model/dense_14/BiasAdd/ReadVariableOp2R
'model/dense_14/Tensordot/ReadVariableOp'model/dense_14/Tensordot/ReadVariableOp2N
%model/dense_15/BiasAdd/ReadVariableOp%model/dense_15/BiasAdd/ReadVariableOp2R
'model/dense_15/Tensordot/ReadVariableOp'model/dense_15/Tensordot/ReadVariableOp2N
%model/dense_16/BiasAdd/ReadVariableOp%model/dense_16/BiasAdd/ReadVariableOp2R
'model/dense_16/Tensordot/ReadVariableOp'model/dense_16/Tensordot/ReadVariableOp2N
%model/dense_17/BiasAdd/ReadVariableOp%model/dense_17/BiasAdd/ReadVariableOp2R
'model/dense_17/Tensordot/ReadVariableOp'model/dense_17/Tensordot/ReadVariableOp2N
%model/dense_18/BiasAdd/ReadVariableOp%model/dense_18/BiasAdd/ReadVariableOp2R
'model/dense_18/Tensordot/ReadVariableOp'model/dense_18/Tensordot/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2P
&model/dense_2/Tensordot/ReadVariableOp&model/dense_2/Tensordot/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2P
&model/dense_3/Tensordot/ReadVariableOp&model/dense_3/Tensordot/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2P
&model/dense_4/Tensordot/ReadVariableOp&model/dense_4/Tensordot/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2P
&model/dense_5/Tensordot/ReadVariableOp&model/dense_5/Tensordot/ReadVariableOp2L
$model/dense_6/BiasAdd/ReadVariableOp$model/dense_6/BiasAdd/ReadVariableOp2P
&model/dense_6/Tensordot/ReadVariableOp&model/dense_6/Tensordot/ReadVariableOp2L
$model/dense_7/BiasAdd/ReadVariableOp$model/dense_7/BiasAdd/ReadVariableOp2P
&model/dense_7/Tensordot/ReadVariableOp&model/dense_7/Tensordot/ReadVariableOp2L
$model/dense_8/BiasAdd/ReadVariableOp$model/dense_8/BiasAdd/ReadVariableOp2P
&model/dense_8/Tensordot/ReadVariableOp&model/dense_8/Tensordot/ReadVariableOp2L
$model/dense_9/BiasAdd/ReadVariableOp$model/dense_9/BiasAdd/ReadVariableOp2P
&model/dense_9/Tensordot/ReadVariableOp&model/dense_9/Tensordot/ReadVariableOp2h
2model/layer_normalization/batchnorm/ReadVariableOp2model/layer_normalization/batchnorm/ReadVariableOp2p
6model/layer_normalization/batchnorm/mul/ReadVariableOp6model/layer_normalization/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_1/batchnorm/ReadVariableOp4model/layer_normalization_1/batchnorm/ReadVariableOp2t
8model/layer_normalization_1/batchnorm/mul/ReadVariableOp8model/layer_normalization_1/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_2/batchnorm/ReadVariableOp4model/layer_normalization_2/batchnorm/ReadVariableOp2t
8model/layer_normalization_2/batchnorm/mul/ReadVariableOp8model/layer_normalization_2/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_3/batchnorm/ReadVariableOp4model/layer_normalization_3/batchnorm/ReadVariableOp2t
8model/layer_normalization_3/batchnorm/mul/ReadVariableOp8model/layer_normalization_3/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_4/batchnorm/ReadVariableOp4model/layer_normalization_4/batchnorm/ReadVariableOp2t
8model/layer_normalization_4/batchnorm/mul/ReadVariableOp8model/layer_normalization_4/batchnorm/mul/ReadVariableOp2l
4model/layer_normalization_5/batchnorm/ReadVariableOp4model/layer_normalization_5/batchnorm/ReadVariableOp2t
8model/layer_normalization_5/batchnorm/mul/ReadVariableOp8model/layer_normalization_5/batchnorm/mul/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: :($
"
_output_shapes
:1

b
)__inference_dropout_5_layer_call_fn_79845

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_75360s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ122
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_1_layer_call_and_return_conditional_losses_74176

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
í

N__inference_layer_normalization_layer_call_and_return_conditional_losses_78868

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¾´
­(
@__inference_model_layer_call_and_return_conditional_losses_77818

inputs
tf_math_multiply_mul_y 
tf___operators___add_addv2_yG
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:;
)dense_2_tensordot_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:;
)dense_3_tensordot_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:<
)dense_5_tensordot_readvariableop_resource:	6
'dense_5_biasadd_readvariableop_resource:	<
)dense_6_tensordot_readvariableop_resource:	5
'dense_6_biasadd_readvariableop_resource:I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:E
7layer_normalization_2_batchnorm_readvariableop_resource:;
)dense_8_tensordot_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:;
)dense_7_tensordot_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:;
)dense_9_tensordot_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:<
*dense_10_tensordot_readvariableop_resource:6
(dense_10_biasadd_readvariableop_resource:I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:E
7layer_normalization_3_batchnorm_readvariableop_resource:=
*dense_11_tensordot_readvariableop_resource:	7
(dense_11_biasadd_readvariableop_resource:	=
*dense_12_tensordot_readvariableop_resource:	6
(dense_12_biasadd_readvariableop_resource:I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:E
7layer_normalization_4_batchnorm_readvariableop_resource:<
*dense_14_tensordot_readvariableop_resource:6
(dense_14_biasadd_readvariableop_resource:<
*dense_13_tensordot_readvariableop_resource:6
(dense_13_biasadd_readvariableop_resource:<
*dense_15_tensordot_readvariableop_resource:6
(dense_15_biasadd_readvariableop_resource:<
*dense_16_tensordot_readvariableop_resource:6
(dense_16_biasadd_readvariableop_resource:I
;layer_normalization_5_batchnorm_mul_readvariableop_resource:E
7layer_normalization_5_batchnorm_readvariableop_resource:=
*dense_17_tensordot_readvariableop_resource:	7
(dense_17_biasadd_readvariableop_resource:	=
*dense_18_tensordot_readvariableop_resource:	6
(dense_18_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	
3
%dense_biasadd_readvariableop_resource:

identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp¢dense_10/BiasAdd/ReadVariableOp¢!dense_10/Tensordot/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢!dense_11/Tensordot/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢!dense_12/Tensordot/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢!dense_13/Tensordot/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢!dense_14/Tensordot/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢!dense_15/Tensordot/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢!dense_16/Tensordot/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢!dense_17/Tensordot/ReadVariableOp¢dense_18/BiasAdd/ReadVariableOp¢!dense_18/Tensordot/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢ dense_3/Tensordot/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢ dense_5/Tensordot/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢ dense_6/Tensordot/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢ dense_7/Tensordot/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢ dense_8/Tensordot/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢ dense_9/Tensordot/ReadVariableOp¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢.layer_normalization_2/batchnorm/ReadVariableOp¢2layer_normalization_2/batchnorm/mul/ReadVariableOp¢.layer_normalization_3/batchnorm/ReadVariableOp¢2layer_normalization_3/batchnorm/mul/ReadVariableOp¢.layer_normalization_4/batchnorm/ReadVariableOp¢2layer_normalization_4/batchnorm/mul/ReadVariableOp¢.layer_normalization_5/batchnorm/ReadVariableOp¢2layer_normalization_5/batchnorm/mul/ReadVariableOpÕ
,tf.image.extract_patches/ExtractImagePatchesExtractImagePatchesinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksizes
*
paddingVALID*
rates
*
strides
s
reshape/ShapeShape6tf.image.extract_patches/ExtractImagePatches:patches:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿY
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :¯
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:¨
reshape/ReshapeReshape6tf.image.extract_patches/ExtractImagePatches:patches:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.math.multiply/MulMulreshape/Reshape:output:0tf_math_multiply_mul_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
dropout/IdentityIdentitytf.__operators__.add/AddV2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ç
 layer_normalization/moments/meanMeandropout/Identity:output:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Æ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout/Identity:output:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ç
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75½
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Á
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
#layer_normalization/batchnorm/mul_1Muldropout/Identity:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1²
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0½
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1²
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
dense_2/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ª
dense_2/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
dense_1/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ª
dense_1/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1W
reshape_1/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Û
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_1/ReshapeReshapedense_2/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Y
reshape_1/Shape_1Shapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_slice_1StridedSlicereshape_1/Shape_1:output:0(reshape_1/strided_slice_1/stack:output:0*reshape_1/strided_slice_1/stack_1:output:0*reshape_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_1/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_1/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_1/Reshape_1/shapePack"reshape_1/strided_slice_1:output:0$reshape_1/Reshape_1/shape/1:output:0$reshape_1/Reshape_1/shape/2:output:0$reshape_1/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_1/Reshape_1Reshapedense_1/BiasAdd:output:0"reshape_1/Reshape_1/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ·
"tf.compat.v1.transpose_1/transpose	Transposereshape_1/Reshape:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.compat.v1.shape/ShapeShape&tf.compat.v1.transpose_1/transpose:y:0*
T0*
_output_shapes
:
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
tf.cast/CastCast/tf.__operators__.getitem/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             µ
 tf.compat.v1.transpose/transpose	Transposereshape_1/Reshape_1:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
dense_3/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ª
dense_3/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1½
tf.linalg.matmul/MatMulBatchMatMulV2$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(L
tf.math.sqrt/SqrtSqrttf.cast/Cast:y:0*
T0*
_output_shapes
: Y
reshape_1/Shape_2Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_slice_2StridedSlicereshape_1/Shape_2:output:0(reshape_1/strided_slice_2/stack:output:0*reshape_1/strided_slice_2/stack_1:output:0*reshape_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_1/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_1/Reshape_2/shapePack"reshape_1/strided_slice_2:output:0$reshape_1/Reshape_2/shape/1:output:0$reshape_1/Reshape_2/shape/2:output:0$reshape_1/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_1/Reshape_2Reshapedense_3/BiasAdd:output:0"reshape_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.math.truediv/truedivRealDiv tf.linalg.matmul/MatMul:output:0tf.math.sqrt/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11w
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¹
"tf.compat.v1.transpose_2/transpose	Transposereshape_1/Reshape_2:output:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
tf.linalg.matmul_1/MatMulBatchMatMulV2tf.nn.softmax/Softmax:softmax:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_3/transpose	Transpose"tf.linalg.matmul_1/MatMul:output:00tf.compat.v1.transpose_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1e
reshape_2/ShapeShape&tf.compat.v1.transpose_3/transpose:y:0*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :·
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_2/ReshapeReshape&tf.compat.v1.transpose_3/transpose:y:0 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       a
dense_4/Tensordot/ShapeShapereshape_2/Reshape:output:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposereshape_2/Reshape:output:0!dense_4/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
dropout_1/IdentityIdentitydense_4/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¡
tf.__operators__.add_1/AddV2AddV2dropout_1/Identity:output:0'layer_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ò
"layer_normalization_1/moments/meanMean tf.__operators__.add_1/AddV2:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ñ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_1/AddV2:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ç
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
%layer_normalization_1/batchnorm/mul_1Mul tf.__operators__.add_1/AddV2:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ã
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0`
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_5/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
dense_5/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
tf.nn.gelu/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
tf.nn.gelu/Gelu/mulMultf.nn.gelu/Gelu/mul/x:output:0dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1[
tf.nn.gelu/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
tf.nn.gelu/Gelu/truedivRealDivdense_5/BiasAdd:output:0tf.nn.gelu/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.nn.gelu/Gelu/ErfErftf.nn.gelu/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
tf.nn.gelu/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu/Gelu/addAddV2tf.nn.gelu/Gelu/add/x:output:0tf.nn.gelu/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu/Gelu/mul_1Multf.nn.gelu/Gelu/mul:z:0tf.nn.gelu/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0`
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       `
dense_6/Tensordot/ShapeShapetf.nn.gelu/Gelu/mul_1:z:0*
T0*
_output_shapes
:a
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_6/Tensordot/transpose	Transposetf.nn.gelu/Gelu/mul_1:z:0!dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
dropout_2/IdentityIdentitydense_6/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
tf.__operators__.add_2/AddV2AddV2dropout_2/Identity:output:0)layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ò
"layer_normalization_2/moments/meanMean tf.__operators__.add_2/AddV2:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ñ
/layer_normalization_2/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_2/AddV2:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ç
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
%layer_normalization_2/batchnorm/mul_1Mul tf.__operators__.add_2/AddV2:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ã
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_8/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
dense_8/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0!dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_7/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
dense_7/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1W
reshape_3/ShapeShapedense_8/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Û
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_3/ReshapeReshapedense_8/BiasAdd:output:0 reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Y
reshape_3/Shape_1Shapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_3/strided_slice_1StridedSlicereshape_3/Shape_1:output:0(reshape_3/strided_slice_1/stack:output:0*reshape_3/strided_slice_1/stack_1:output:0*reshape_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_3/Reshape_1/shapePack"reshape_3/strided_slice_1:output:0$reshape_3/Reshape_1/shape/1:output:0$reshape_3/Reshape_1/shape/2:output:0$reshape_3/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_3/Reshape_1Reshapedense_7/BiasAdd:output:0"reshape_3/Reshape_1/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_5/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ·
"tf.compat.v1.transpose_5/transpose	Transposereshape_3/Reshape:output:00tf.compat.v1.transpose_5/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_1/ShapeShape&tf.compat.v1.transpose_5/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_1/CastCast1tf.__operators__.getitem_1/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¹
"tf.compat.v1.transpose_4/transpose	Transposereshape_3/Reshape_1:output:00tf.compat.v1.transpose_4/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_9/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
dense_9/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Á
tf.linalg.matmul_2/MatMulBatchMatMulV2&tf.compat.v1.transpose_4/transpose:y:0&tf.compat.v1.transpose_5/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_1/SqrtSqrttf.cast_1/Cast:y:0*
T0*
_output_shapes
: Y
reshape_3/Shape_2Shapedense_9/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_3/strided_slice_2StridedSlicereshape_3/Shape_2:output:0(reshape_3/strided_slice_2/stack:output:0*reshape_3/strided_slice_2/stack_1:output:0*reshape_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_3/Reshape_2/shapePack"reshape_3/strided_slice_2:output:0$reshape_3/Reshape_2/shape/1:output:0$reshape_3/Reshape_2/shape/2:output:0$reshape_3/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_3/Reshape_2Reshapedense_9/BiasAdd:output:0"reshape_3/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.math.truediv_1/truedivRealDiv"tf.linalg.matmul_2/MatMul:output:0tf.math.sqrt_1/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_6/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¹
"tf.compat.v1.transpose_6/transpose	Transposereshape_3/Reshape_2:output:00tf.compat.v1.transpose_6/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¯
tf.linalg.matmul_3/MatMulBatchMatMulV2!tf.nn.softmax_1/Softmax:softmax:0&tf.compat.v1.transpose_6/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_7/transpose	Transpose"tf.linalg.matmul_3/MatMul:output:00tf.compat.v1.transpose_7/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1e
reshape_4/ShapeShape&tf.compat.v1.transpose_7/transpose:y:0*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :·
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_4/ReshapeReshape&tf.compat.v1.transpose_7/transpose:y:0 reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_10/Tensordot/ShapeShapereshape_4/Reshape:output:0*
T0*
_output_shapes
:b
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_10/Tensordot/transpose	Transposereshape_4/Reshape:output:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1o
dropout_3/IdentityIdentitydense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
tf.__operators__.add_3/AddV2AddV2dropout_3/Identity:output:0)layer_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ò
"layer_normalization_3/moments/meanMean tf.__operators__.add_3/AddV2:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ñ
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_3/AddV2:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ç
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
%layer_normalization_3/batchnorm/mul_1Mul tf.__operators__.add_3/AddV2:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ã
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0a
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
dense_11/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:b
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
dense_11/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0"dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
tf.nn.gelu_1/Gelu/mulMul tf.nn.gelu_1/Gelu/mul/x:output:0dense_11/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
tf.nn.gelu_1/Gelu/truedivRealDivdense_11/BiasAdd:output:0!tf.nn.gelu_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_1/Gelu/ErfErftf.nn.gelu_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_1/Gelu/addAddV2 tf.nn.gelu_1/Gelu/add/x:output:0tf.nn.gelu_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_1/Gelu/mul_1Multf.nn.gelu_1/Gelu/mul:z:0tf.nn.gelu_1/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_12/Tensordot/ReadVariableOpReadVariableOp*dense_12_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0a
dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_12/Tensordot/ShapeShapetf.nn.gelu_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:b
 dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_12/Tensordot/GatherV2GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/free:output:0)dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_12/Tensordot/GatherV2_1GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/axes:output:0+dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_12/Tensordot/ProdProd$dense_12/Tensordot/GatherV2:output:0!dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_12/Tensordot/Prod_1Prod&dense_12/Tensordot/GatherV2_1:output:0#dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_12/Tensordot/concatConcatV2 dense_12/Tensordot/free:output:0 dense_12/Tensordot/axes:output:0'dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_12/Tensordot/stackPack dense_12/Tensordot/Prod:output:0"dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¡
dense_12/Tensordot/transpose	Transposetf.nn.gelu_1/Gelu/mul_1:z:0"dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_12/Tensordot/ReshapeReshape dense_12/Tensordot/transpose:y:0!dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_12/Tensordot/MatMulMatMul#dense_12/Tensordot/Reshape:output:0)dense_12/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_12/Tensordot/concat_1ConcatV2$dense_12/Tensordot/GatherV2:output:0#dense_12/Tensordot/Const_2:output:0)dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_12/TensordotReshape#dense_12/Tensordot/MatMul:product:0$dense_12/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_12/BiasAddBiasAdddense_12/Tensordot:output:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1o
dropout_4/IdentityIdentitydense_12/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
tf.__operators__.add_4/AddV2AddV2dropout_4/Identity:output:0)layer_normalization_3/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ò
"layer_normalization_4/moments/meanMean tf.__operators__.add_4/AddV2:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ñ
/layer_normalization_4/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_4/AddV2:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(j
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ç
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
%layer_normalization_4/batchnorm/mul_1Mul tf.__operators__.add_4/AddV2:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ã
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_14/Tensordot/ReadVariableOpReadVariableOp*dense_14_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
dense_14/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:b
 dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_14/Tensordot/GatherV2GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/free:output:0)dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_14/Tensordot/GatherV2_1GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/axes:output:0+dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/ProdProd$dense_14/Tensordot/GatherV2:output:0!dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/Prod_1Prod&dense_14/Tensordot/GatherV2_1:output:0#dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_14/Tensordot/concatConcatV2 dense_14/Tensordot/free:output:0 dense_14/Tensordot/axes:output:0'dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/stackPack dense_14/Tensordot/Prod:output:0"dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
dense_14/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0"dense_14/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_14/Tensordot/ReshapeReshape dense_14/Tensordot/transpose:y:0!dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/MatMulMatMul#dense_14/Tensordot/Reshape:output:0)dense_14/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_14/Tensordot/concat_1ConcatV2$dense_14/Tensordot/GatherV2:output:0#dense_14/Tensordot/Const_2:output:0)dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/TensordotReshape#dense_14/Tensordot/MatMul:product:0$dense_14/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/Tensordot:output:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_13/Tensordot/ReadVariableOpReadVariableOp*dense_13_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
dense_13/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:b
 dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_13/Tensordot/GatherV2GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/free:output:0)dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_13/Tensordot/GatherV2_1GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/axes:output:0+dense_13/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_13/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_13/Tensordot/ProdProd$dense_13/Tensordot/GatherV2:output:0!dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_13/Tensordot/Prod_1Prod&dense_13/Tensordot/GatherV2_1:output:0#dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_13/Tensordot/concatConcatV2 dense_13/Tensordot/free:output:0 dense_13/Tensordot/axes:output:0'dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_13/Tensordot/stackPack dense_13/Tensordot/Prod:output:0"dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
dense_13/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0"dense_13/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_13/Tensordot/ReshapeReshape dense_13/Tensordot/transpose:y:0!dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_13/Tensordot/MatMulMatMul#dense_13/Tensordot/Reshape:output:0)dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_13/Tensordot/concat_1ConcatV2$dense_13/Tensordot/GatherV2:output:0#dense_13/Tensordot/Const_2:output:0)dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_13/TensordotReshape#dense_13/Tensordot/MatMul:product:0$dense_13/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_13/BiasAddBiasAdddense_13/Tensordot:output:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1X
reshape_5/ShapeShapedense_14/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_5/strided_sliceStridedSlicereshape_5/Shape:output:0&reshape_5/strided_slice/stack:output:0(reshape_5/strided_slice/stack_1:output:0(reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Û
reshape_5/Reshape/shapePack reshape_5/strided_slice:output:0"reshape_5/Reshape/shape/1:output:0"reshape_5/Reshape/shape/2:output:0"reshape_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_5/ReshapeReshapedense_14/BiasAdd:output:0 reshape_5/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
reshape_5/Shape_1Shapedense_13/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_5/strided_slice_1StridedSlicereshape_5/Shape_1:output:0(reshape_5/strided_slice_1/stack:output:0*reshape_5/strided_slice_1/stack_1:output:0*reshape_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_5/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_5/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_5/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_5/Reshape_1/shapePack"reshape_5/strided_slice_1:output:0$reshape_5/Reshape_1/shape/1:output:0$reshape_5/Reshape_1/shape/2:output:0$reshape_5/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_5/Reshape_1Reshapedense_13/BiasAdd:output:0"reshape_5/Reshape_1/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ·
"tf.compat.v1.transpose_9/transpose	Transposereshape_5/Reshape:output:00tf.compat.v1.transpose_9/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_2/ShapeShape&tf.compat.v1.transpose_9/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_2/strided_sliceStridedSlice#tf.compat.v1.shape_2/Shape:output:07tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_2/CastCast1tf.__operators__.getitem_2/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¹
"tf.compat.v1.transpose_8/transpose	Transposereshape_5/Reshape_1:output:00tf.compat.v1.transpose_8/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_15/Tensordot/ReadVariableOpReadVariableOp*dense_15_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_15/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_15/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
dense_15/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:b
 dense_15/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_15/Tensordot/GatherV2GatherV2!dense_15/Tensordot/Shape:output:0 dense_15/Tensordot/free:output:0)dense_15/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_15/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_15/Tensordot/GatherV2_1GatherV2!dense_15/Tensordot/Shape:output:0 dense_15/Tensordot/axes:output:0+dense_15/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_15/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_15/Tensordot/ProdProd$dense_15/Tensordot/GatherV2:output:0!dense_15/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_15/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_15/Tensordot/Prod_1Prod&dense_15/Tensordot/GatherV2_1:output:0#dense_15/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_15/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_15/Tensordot/concatConcatV2 dense_15/Tensordot/free:output:0 dense_15/Tensordot/axes:output:0'dense_15/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_15/Tensordot/stackPack dense_15/Tensordot/Prod:output:0"dense_15/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
dense_15/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0"dense_15/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_15/Tensordot/ReshapeReshape dense_15/Tensordot/transpose:y:0!dense_15/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_15/Tensordot/MatMulMatMul#dense_15/Tensordot/Reshape:output:0)dense_15/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_15/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_15/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_15/Tensordot/concat_1ConcatV2$dense_15/Tensordot/GatherV2:output:0#dense_15/Tensordot/Const_2:output:0)dense_15/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_15/TensordotReshape#dense_15/Tensordot/MatMul:product:0$dense_15/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_15/BiasAddBiasAdddense_15/Tensordot:output:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Á
tf.linalg.matmul_4/MatMulBatchMatMulV2&tf.compat.v1.transpose_8/transpose:y:0&tf.compat.v1.transpose_9/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_2/SqrtSqrttf.cast_2/Cast:y:0*
T0*
_output_shapes
: Z
reshape_5/Shape_2Shapedense_15/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_5/strided_slice_2StridedSlicereshape_5/Shape_2:output:0(reshape_5/strided_slice_2/stack:output:0*reshape_5/strided_slice_2/stack_1:output:0*reshape_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_5/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_5/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_5/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_5/Reshape_2/shapePack"reshape_5/strided_slice_2:output:0$reshape_5/Reshape_2/shape/1:output:0$reshape_5/Reshape_2/shape/2:output:0$reshape_5/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_5/Reshape_2Reshapedense_15/BiasAdd:output:0"reshape_5/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.math.truediv_2/truedivRealDiv"tf.linalg.matmul_4/MatMul:output:0tf.math.sqrt_2/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_2/SoftmaxSoftmaxtf.math.truediv_2/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
(tf.compat.v1.transpose_10/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             »
#tf.compat.v1.transpose_10/transpose	Transposereshape_5/Reshape_2:output:01tf.compat.v1.transpose_10/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1°
tf.linalg.matmul_5/MatMulBatchMatMulV2!tf.nn.softmax_2/Softmax:softmax:0'tf.compat.v1.transpose_10/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
(tf.compat.v1.transpose_11/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
#tf.compat.v1.transpose_11/transpose	Transpose"tf.linalg.matmul_5/MatMul:output:01tf.compat.v1.transpose_11/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
reshape_6/ShapeShape'tf.compat.v1.transpose_11/transpose:y:0*
T0*
_output_shapes
:g
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_6/strided_sliceStridedSlicereshape_6/Shape:output:0&reshape_6/strided_slice/stack:output:0(reshape_6/strided_slice/stack_1:output:0(reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :·
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0"reshape_6/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_6/ReshapeReshape'tf.compat.v1.transpose_11/transpose:y:0 reshape_6/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_16/Tensordot/ReadVariableOpReadVariableOp*dense_16_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_16/Tensordot/ShapeShapereshape_6/Reshape:output:0*
T0*
_output_shapes
:b
 dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_16/Tensordot/GatherV2GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/free:output:0)dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_16/Tensordot/GatherV2_1GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/axes:output:0+dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_16/Tensordot/ProdProd$dense_16/Tensordot/GatherV2:output:0!dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_16/Tensordot/Prod_1Prod&dense_16/Tensordot/GatherV2_1:output:0#dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_16/Tensordot/concatConcatV2 dense_16/Tensordot/free:output:0 dense_16/Tensordot/axes:output:0'dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_16/Tensordot/stackPack dense_16/Tensordot/Prod:output:0"dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_16/Tensordot/transpose	Transposereshape_6/Reshape:output:0"dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_16/Tensordot/ReshapeReshape dense_16/Tensordot/transpose:y:0!dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_16/Tensordot/MatMulMatMul#dense_16/Tensordot/Reshape:output:0)dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_16/Tensordot/concat_1ConcatV2$dense_16/Tensordot/GatherV2:output:0#dense_16/Tensordot/Const_2:output:0)dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_16/TensordotReshape#dense_16/Tensordot/MatMul:product:0$dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_16/BiasAddBiasAdddense_16/Tensordot:output:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1o
dropout_5/IdentityIdentitydense_16/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
tf.__operators__.add_5/AddV2AddV2dropout_5/Identity:output:0)layer_normalization_4/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ò
"layer_normalization_5/moments/meanMean tf.__operators__.add_5/AddV2:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ñ
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_5/AddV2:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(j
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ç
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
%layer_normalization_5/batchnorm/mul_1Mul tf.__operators__.add_5/AddV2:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ã
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_17/Tensordot/ReadVariableOpReadVariableOp*dense_17_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0a
dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
dense_17/Tensordot/ShapeShape)layer_normalization_5/batchnorm/add_1:z:0*
T0*
_output_shapes
:b
 dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_17/Tensordot/GatherV2GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/free:output:0)dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_17/Tensordot/GatherV2_1GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/axes:output:0+dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_17/Tensordot/ProdProd$dense_17/Tensordot/GatherV2:output:0!dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_17/Tensordot/Prod_1Prod&dense_17/Tensordot/GatherV2_1:output:0#dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_17/Tensordot/concatConcatV2 dense_17/Tensordot/free:output:0 dense_17/Tensordot/axes:output:0'dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_17/Tensordot/stackPack dense_17/Tensordot/Prod:output:0"dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
dense_17/Tensordot/transpose	Transpose)layer_normalization_5/batchnorm/add_1:z:0"dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_17/Tensordot/ReshapeReshape dense_17/Tensordot/transpose:y:0!dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
dense_17/Tensordot/MatMulMatMul#dense_17/Tensordot/Reshape:output:0)dense_17/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_17/Tensordot/concat_1ConcatV2$dense_17/Tensordot/GatherV2:output:0#dense_17/Tensordot/Const_2:output:0)dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_17/TensordotReshape#dense_17/Tensordot/MatMul:product:0$dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_17/BiasAddBiasAdddense_17/Tensordot:output:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
tf.nn.gelu_2/Gelu/mulMul tf.nn.gelu_2/Gelu/mul/x:output:0dense_17/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
tf.nn.gelu_2/Gelu/truedivRealDivdense_17/BiasAdd:output:0!tf.nn.gelu_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_2/Gelu/ErfErftf.nn.gelu_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_2/Gelu/addAddV2 tf.nn.gelu_2/Gelu/add/x:output:0tf.nn.gelu_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_2/Gelu/mul_1Multf.nn.gelu_2/Gelu/mul:z:0tf.nn.gelu_2/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_18/Tensordot/ReadVariableOpReadVariableOp*dense_18_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0a
dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_18/Tensordot/ShapeShapetf.nn.gelu_2/Gelu/mul_1:z:0*
T0*
_output_shapes
:b
 dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_18/Tensordot/GatherV2GatherV2!dense_18/Tensordot/Shape:output:0 dense_18/Tensordot/free:output:0)dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_18/Tensordot/GatherV2_1GatherV2!dense_18/Tensordot/Shape:output:0 dense_18/Tensordot/axes:output:0+dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_18/Tensordot/ProdProd$dense_18/Tensordot/GatherV2:output:0!dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_18/Tensordot/Prod_1Prod&dense_18/Tensordot/GatherV2_1:output:0#dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_18/Tensordot/concatConcatV2 dense_18/Tensordot/free:output:0 dense_18/Tensordot/axes:output:0'dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_18/Tensordot/stackPack dense_18/Tensordot/Prod:output:0"dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¡
dense_18/Tensordot/transpose	Transposetf.nn.gelu_2/Gelu/mul_1:z:0"dense_18/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_18/Tensordot/ReshapeReshape dense_18/Tensordot/transpose:y:0!dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_18/Tensordot/MatMulMatMul#dense_18/Tensordot/Reshape:output:0)dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_18/Tensordot/concat_1ConcatV2$dense_18/Tensordot/GatherV2:output:0#dense_18/Tensordot/Const_2:output:0)dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_18/TensordotReshape#dense_18/Tensordot/MatMul:product:0$dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_18/BiasAddBiasAdddense_18/Tensordot:output:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1o
dropout_6/IdentityIdentitydense_18/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
tf.__operators__.add_6/AddV2AddV2dropout_6/Identity:output:0)layer_normalization_5/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
flatten/ReshapeReshape tf.__operators__.add_6/AddV2:z:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
softmax/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp"^dense_12/Tensordot/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp"^dense_13/Tensordot/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp"^dense_14/Tensordot/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp"^dense_15/Tensordot/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp"^dense_16/Tensordot/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp"^dense_17/Tensordot/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp"^dense_18/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2F
!dense_12/Tensordot/ReadVariableOp!dense_12/Tensordot/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2F
!dense_13/Tensordot/ReadVariableOp!dense_13/Tensordot/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/Tensordot/ReadVariableOp!dense_14/Tensordot/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2F
!dense_15/Tensordot/ReadVariableOp!dense_15/Tensordot/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2F
!dense_16/Tensordot/ReadVariableOp!dense_16/Tensordot/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2F
!dense_17/Tensordot/ReadVariableOp!dense_17/Tensordot/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2F
!dense_18/Tensordot/ReadVariableOp!dense_18/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :($
"
_output_shapes
:1
Ô

'__inference_dense_6_layer_call_fn_79167

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_74416s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_75123

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Èµ
ô
@__inference_model_layer_call_and_return_conditional_losses_76084

inputs
tf_math_multiply_mul_y 
tf___operators___add_addv2_y'
layer_normalization_75851:'
layer_normalization_75853:
dense_2_75856:
dense_2_75858:
dense_1_75861:
dense_1_75863:
dense_3_75878:
dense_3_75880:
dense_4_75894:
dense_4_75896:)
layer_normalization_1_75901:)
layer_normalization_1_75903: 
dense_5_75906:	
dense_5_75908:	 
dense_6_75919:	
dense_6_75921:)
layer_normalization_2_75926:)
layer_normalization_2_75928:
dense_8_75931:
dense_8_75933:
dense_7_75936:
dense_7_75938:
dense_9_75953:
dense_9_75955: 
dense_10_75969:
dense_10_75971:)
layer_normalization_3_75976:)
layer_normalization_3_75978:!
dense_11_75981:	
dense_11_75983:	!
dense_12_75994:	
dense_12_75996:)
layer_normalization_4_76001:)
layer_normalization_4_76003: 
dense_14_76006:
dense_14_76008: 
dense_13_76011:
dense_13_76013: 
dense_15_76028:
dense_15_76030: 
dense_16_76044:
dense_16_76046:)
layer_normalization_5_76051:)
layer_normalization_5_76053:!
dense_17_76056:	
dense_17_76058:	!
dense_18_76069:	
dense_18_76071:
dense_76077:	

dense_76079:

identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢!dropout_6/StatefulPartitionedCall¢+layer_normalization/StatefulPartitionedCall¢-layer_normalization_1/StatefulPartitionedCall¢-layer_normalization_2/StatefulPartitionedCall¢-layer_normalization_3/StatefulPartitionedCall¢-layer_normalization_4/StatefulPartitionedCall¢-layer_normalization_5/StatefulPartitionedCallÕ
,tf.image.extract_patches/ExtractImagePatchesExtractImagePatchesinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksizes
*
paddingVALID*
rates
*
strides
ì
reshape/PartitionedCallPartitionedCall6tf.image.extract_patches/ExtractImagePatches:patches:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_74069
tf.math.multiply/MulMul reshape/PartitionedCall:output:0tf_math_multiply_mul_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ä
dropout/StatefulPartitionedCallStatefulPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_75721Â
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0layer_normalization_75851layer_normalization_75853*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_74104
dense_2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_2_75856dense_2_75858*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_74140
dense_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_1_75861dense_1_75863*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_74176æ
reshape_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196è
reshape_1/PartitionedCall_1PartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_1/transpose	Transpose"reshape_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.compat.v1.shape/ShapeShape&tf.compat.v1.transpose_1/transpose:y:0*
T0*
_output_shapes
:
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
tf.cast/CastCast/tf.__operators__.getitem/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ½
 tf.compat.v1.transpose/transpose	Transpose$reshape_1/PartitionedCall_1:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_3/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_3_75878dense_3_75880*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_74239½
tf.linalg.matmul/MatMulBatchMatMulV2$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(L
tf.math.sqrt/SqrtSqrttf.cast/Cast:y:0*
T0*
_output_shapes
: è
reshape_1/PartitionedCall_2PartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196
tf.math.truediv/truedivRealDiv tf.linalg.matmul/MatMul:output:0tf.math.sqrt/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11w
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_2/transpose	Transpose$reshape_1/PartitionedCall_2:output:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
tf.linalg.matmul_1/MatMulBatchMatMulV2tf.nn.softmax/Softmax:softmax:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_3/transpose	Transpose"tf.linalg.matmul_1/MatMul:output:00tf.compat.v1.transpose_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1à
reshape_2/PartitionedCallPartitionedCall&tf.compat.v1.transpose_3/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_74268
dense_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0dense_4_75894dense_4_75896*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_74300
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_75636½
tf.__operators__.add_1/AddV2AddV2*dropout_1/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_1_75901layer_normalization_1_75903*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_74336¡
dense_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_5_75906dense_5_75908*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_74372Z
tf.nn.gelu/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
tf.nn.gelu/Gelu/mulMultf.nn.gelu/Gelu/mul/x:output:0(dense_5/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1[
tf.nn.gelu/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?¤
tf.nn.gelu/Gelu/truedivRealDiv(dense_5/StatefulPartitionedCall:output:0tf.nn.gelu/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.nn.gelu/Gelu/ErfErftf.nn.gelu/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
tf.nn.gelu/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu/Gelu/addAddV2tf.nn.gelu/Gelu/add/x:output:0tf.nn.gelu/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu/Gelu/mul_1Multf.nn.gelu/Gelu/mul:z:0tf.nn.gelu/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_6/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu/Gelu/mul_1:z:0dense_6_75919dense_6_75921*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_74416
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_75583¿
tf.__operators__.add_2/AddV2AddV2*dropout_2/StatefulPartitionedCall:output:06layer_normalization_1/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_2_75926layer_normalization_2_75928*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_74452 
dense_8/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_8_75931dense_8_75933*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_74488 
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_7_75936dense_7_75938*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_74524æ
reshape_3/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544è
reshape_3/PartitionedCall_1PartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544
'tf.compat.v1.transpose_5/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_5/transpose	Transpose"reshape_3/PartitionedCall:output:00tf.compat.v1.transpose_5/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_1/ShapeShape&tf.compat.v1.transpose_5/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_1/CastCast1tf.__operators__.getitem_1/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_4/transpose	Transpose$reshape_3/PartitionedCall_1:output:00tf.compat.v1.transpose_4/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1 
dense_9/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_9_75953dense_9_75955*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_74587Á
tf.linalg.matmul_2/MatMulBatchMatMulV2&tf.compat.v1.transpose_4/transpose:y:0&tf.compat.v1.transpose_5/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_1/SqrtSqrttf.cast_1/Cast:y:0*
T0*
_output_shapes
: è
reshape_3/PartitionedCall_2PartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544
tf.math.truediv_1/truedivRealDiv"tf.linalg.matmul_2/MatMul:output:0tf.math.sqrt_1/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_6/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_6/transpose	Transpose$reshape_3/PartitionedCall_2:output:00tf.compat.v1.transpose_6/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¯
tf.linalg.matmul_3/MatMulBatchMatMulV2!tf.nn.softmax_1/Softmax:softmax:0&tf.compat.v1.transpose_6/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_7/transpose	Transpose"tf.linalg.matmul_3/MatMul:output:00tf.compat.v1.transpose_7/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1à
reshape_4/PartitionedCallPartitionedCall&tf.compat.v1.transpose_7/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_74616
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0dense_10_75969dense_10_75971*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_74648
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall)dense_10/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_75498¿
tf.__operators__.add_3/AddV2AddV2*dropout_3/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_3_75976layer_normalization_3_75978*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_74684¥
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_11_75981dense_11_75983*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_74720\
tf.nn.gelu_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
tf.nn.gelu_1/Gelu/mulMul tf.nn.gelu_1/Gelu/mul/x:output:0)dense_11/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?©
tf.nn.gelu_1/Gelu/truedivRealDiv)dense_11/StatefulPartitionedCall:output:0!tf.nn.gelu_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_1/Gelu/ErfErftf.nn.gelu_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_1/Gelu/addAddV2 tf.nn.gelu_1/Gelu/add/x:output:0tf.nn.gelu_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_1/Gelu/mul_1Multf.nn.gelu_1/Gelu/mul:z:0tf.nn.gelu_1/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_12/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu_1/Gelu/mul_1:z:0dense_12_75994dense_12_75996*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_74764
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_75445¿
tf.__operators__.add_4/AddV2AddV2*dropout_4/StatefulPartitionedCall:output:06layer_normalization_3/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_4_76001layer_normalization_4_76003*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_74800¤
 dense_14/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_14_76006dense_14_76008*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_74836¤
 dense_13/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_13_76011dense_13_76013*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_74872ç
reshape_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892é
reshape_5/PartitionedCall_1PartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892
'tf.compat.v1.transpose_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_9/transpose	Transpose"reshape_5/PartitionedCall:output:00tf.compat.v1.transpose_9/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_2/ShapeShape&tf.compat.v1.transpose_9/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_2/strided_sliceStridedSlice#tf.compat.v1.shape_2/Shape:output:07tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_2/CastCast1tf.__operators__.getitem_2/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_8/transpose	Transpose$reshape_5/PartitionedCall_1:output:00tf.compat.v1.transpose_8/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¤
 dense_15/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_15_76028dense_15_76030*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_74935Á
tf.linalg.matmul_4/MatMulBatchMatMulV2&tf.compat.v1.transpose_8/transpose:y:0&tf.compat.v1.transpose_9/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_2/SqrtSqrttf.cast_2/Cast:y:0*
T0*
_output_shapes
: é
reshape_5/PartitionedCall_2PartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892
tf.math.truediv_2/truedivRealDiv"tf.linalg.matmul_4/MatMul:output:0tf.math.sqrt_2/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_2/SoftmaxSoftmaxtf.math.truediv_2/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
(tf.compat.v1.transpose_10/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ã
#tf.compat.v1.transpose_10/transpose	Transpose$reshape_5/PartitionedCall_2:output:01tf.compat.v1.transpose_10/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1°
tf.linalg.matmul_5/MatMulBatchMatMulV2!tf.nn.softmax_2/Softmax:softmax:0'tf.compat.v1.transpose_10/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
(tf.compat.v1.transpose_11/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
#tf.compat.v1.transpose_11/transpose	Transpose"tf.linalg.matmul_5/MatMul:output:01tf.compat.v1.transpose_11/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1á
reshape_6/PartitionedCallPartitionedCall'tf.compat.v1.transpose_11/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_6_layer_call_and_return_conditional_losses_74964
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0dense_16_76044dense_16_76046*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_74996
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_75360¿
tf.__operators__.add_5/AddV2AddV2*dropout_5/StatefulPartitionedCall:output:06layer_normalization_4/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_5_76051layer_normalization_5_76053*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_75032¥
 dense_17/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0dense_17_76056dense_17_76058*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_75068\
tf.nn.gelu_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
tf.nn.gelu_2/Gelu/mulMul tf.nn.gelu_2/Gelu/mul/x:output:0)dense_17/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?©
tf.nn.gelu_2/Gelu/truedivRealDiv)dense_17/StatefulPartitionedCall:output:0!tf.nn.gelu_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_2/Gelu/ErfErftf.nn.gelu_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_2/Gelu/addAddV2 tf.nn.gelu_2/Gelu/add/x:output:0tf.nn.gelu_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_2/Gelu/mul_1Multf.nn.gelu_2/Gelu/mul:z:0tf.nn.gelu_2/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu_2/Gelu/mul_1:z:0dense_18_76069dense_18_76071*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_75112
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall)dense_18/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_75307¿
tf.__operators__.add_6/AddV2AddV2*dropout_6/StatefulPartitionedCall:output:06layer_normalization_5/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ó
flatten/PartitionedCallPartitionedCall tf.__operators__.add_6/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75132þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_76077dense_76079*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75144Ø
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_75155o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ë	
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :($
"
_output_shapes
:1
É
ù
B__inference_dense_4_layer_call_and_return_conditional_losses_79061

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ê
ú
C__inference_dense_10_layer_call_and_return_conditional_losses_74648

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

`
D__inference_reshape_2_layer_call_and_return_conditional_losses_74268

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_1_layer_call_and_return_conditional_losses_78907

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
é

5__inference_layer_normalization_1_layer_call_fn_79097

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_74336s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_6_layer_call_and_return_conditional_losses_75307

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

C
'__inference_softmax_layer_call_fn_80033

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_75155`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
âª
ú
@__inference_model_layer_call_and_return_conditional_losses_75158

inputs
tf_math_multiply_mul_y 
tf___operators___add_addv2_y'
layer_normalization_74105:'
layer_normalization_74107:
dense_2_74141:
dense_2_74143:
dense_1_74177:
dense_1_74179:
dense_3_74240:
dense_3_74242:
dense_4_74301:
dense_4_74303:)
layer_normalization_1_74337:)
layer_normalization_1_74339: 
dense_5_74373:	
dense_5_74375:	 
dense_6_74417:	
dense_6_74419:)
layer_normalization_2_74453:)
layer_normalization_2_74455:
dense_8_74489:
dense_8_74491:
dense_7_74525:
dense_7_74527:
dense_9_74588:
dense_9_74590: 
dense_10_74649:
dense_10_74651:)
layer_normalization_3_74685:)
layer_normalization_3_74687:!
dense_11_74721:	
dense_11_74723:	!
dense_12_74765:	
dense_12_74767:)
layer_normalization_4_74801:)
layer_normalization_4_74803: 
dense_14_74837:
dense_14_74839: 
dense_13_74873:
dense_13_74875: 
dense_15_74936:
dense_15_74938: 
dense_16_74997:
dense_16_74999:)
layer_normalization_5_75033:)
layer_normalization_5_75035:!
dense_17_75069:	
dense_17_75071:	!
dense_18_75113:	
dense_18_75115:
dense_75145:	

dense_75147:

identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢+layer_normalization/StatefulPartitionedCall¢-layer_normalization_1/StatefulPartitionedCall¢-layer_normalization_2/StatefulPartitionedCall¢-layer_normalization_3/StatefulPartitionedCall¢-layer_normalization_4/StatefulPartitionedCall¢-layer_normalization_5/StatefulPartitionedCallÕ
,tf.image.extract_patches/ExtractImagePatchesExtractImagePatchesinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksizes
*
paddingVALID*
rates
*
strides
ì
reshape/PartitionedCallPartitionedCall6tf.image.extract_patches/ExtractImagePatches:patches:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_74069
tf.math.multiply/MulMul reshape/PartitionedCall:output:0tf_math_multiply_mul_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ô
dropout/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_74080º
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0layer_normalization_74105layer_normalization_74107*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_74104
dense_2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_2_74141dense_2_74143*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_74140
dense_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_1_74177dense_1_74179*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_74176æ
reshape_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196è
reshape_1/PartitionedCall_1PartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_1/transpose	Transpose"reshape_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.compat.v1.shape/ShapeShape&tf.compat.v1.transpose_1/transpose:y:0*
T0*
_output_shapes
:
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
tf.cast/CastCast/tf.__operators__.getitem/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ½
 tf.compat.v1.transpose/transpose	Transpose$reshape_1/PartitionedCall_1:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_3/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_3_74240dense_3_74242*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_74239½
tf.linalg.matmul/MatMulBatchMatMulV2$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(L
tf.math.sqrt/SqrtSqrttf.cast/Cast:y:0*
T0*
_output_shapes
: è
reshape_1/PartitionedCall_2PartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196
tf.math.truediv/truedivRealDiv tf.linalg.matmul/MatMul:output:0tf.math.sqrt/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11w
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_2/transpose	Transpose$reshape_1/PartitionedCall_2:output:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
tf.linalg.matmul_1/MatMulBatchMatMulV2tf.nn.softmax/Softmax:softmax:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_3/transpose	Transpose"tf.linalg.matmul_1/MatMul:output:00tf.compat.v1.transpose_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1à
reshape_2/PartitionedCallPartitionedCall&tf.compat.v1.transpose_3/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_74268
dense_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0dense_4_74301dense_4_74303*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_74300â
dropout_1/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_74311µ
tf.__operators__.add_1/AddV2AddV2"dropout_1/PartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_1_74337layer_normalization_1_74339*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_74336¡
dense_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_5_74373dense_5_74375*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_74372Z
tf.nn.gelu/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
tf.nn.gelu/Gelu/mulMultf.nn.gelu/Gelu/mul/x:output:0(dense_5/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1[
tf.nn.gelu/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?¤
tf.nn.gelu/Gelu/truedivRealDiv(dense_5/StatefulPartitionedCall:output:0tf.nn.gelu/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.nn.gelu/Gelu/ErfErftf.nn.gelu/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
tf.nn.gelu/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu/Gelu/addAddV2tf.nn.gelu/Gelu/add/x:output:0tf.nn.gelu/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu/Gelu/mul_1Multf.nn.gelu/Gelu/mul:z:0tf.nn.gelu/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_6/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu/Gelu/mul_1:z:0dense_6_74417dense_6_74419*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_74416â
dropout_2/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_74427·
tf.__operators__.add_2/AddV2AddV2"dropout_2/PartitionedCall:output:06layer_normalization_1/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_2_74453layer_normalization_2_74455*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_74452 
dense_8/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_8_74489dense_8_74491*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_74488 
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_7_74525dense_7_74527*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_74524æ
reshape_3/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544è
reshape_3/PartitionedCall_1PartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544
'tf.compat.v1.transpose_5/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_5/transpose	Transpose"reshape_3/PartitionedCall:output:00tf.compat.v1.transpose_5/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_1/ShapeShape&tf.compat.v1.transpose_5/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_1/CastCast1tf.__operators__.getitem_1/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_4/transpose	Transpose$reshape_3/PartitionedCall_1:output:00tf.compat.v1.transpose_4/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1 
dense_9/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_9_74588dense_9_74590*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_74587Á
tf.linalg.matmul_2/MatMulBatchMatMulV2&tf.compat.v1.transpose_4/transpose:y:0&tf.compat.v1.transpose_5/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_1/SqrtSqrttf.cast_1/Cast:y:0*
T0*
_output_shapes
: è
reshape_3/PartitionedCall_2PartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544
tf.math.truediv_1/truedivRealDiv"tf.linalg.matmul_2/MatMul:output:0tf.math.sqrt_1/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_6/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_6/transpose	Transpose$reshape_3/PartitionedCall_2:output:00tf.compat.v1.transpose_6/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¯
tf.linalg.matmul_3/MatMulBatchMatMulV2!tf.nn.softmax_1/Softmax:softmax:0&tf.compat.v1.transpose_6/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_7/transpose	Transpose"tf.linalg.matmul_3/MatMul:output:00tf.compat.v1.transpose_7/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1à
reshape_4/PartitionedCallPartitionedCall&tf.compat.v1.transpose_7/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_74616
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0dense_10_74649dense_10_74651*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_74648ã
dropout_3/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_74659·
tf.__operators__.add_3/AddV2AddV2"dropout_3/PartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_3_74685layer_normalization_3_74687*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_74684¥
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_11_74721dense_11_74723*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_74720\
tf.nn.gelu_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
tf.nn.gelu_1/Gelu/mulMul tf.nn.gelu_1/Gelu/mul/x:output:0)dense_11/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?©
tf.nn.gelu_1/Gelu/truedivRealDiv)dense_11/StatefulPartitionedCall:output:0!tf.nn.gelu_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_1/Gelu/ErfErftf.nn.gelu_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_1/Gelu/addAddV2 tf.nn.gelu_1/Gelu/add/x:output:0tf.nn.gelu_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_1/Gelu/mul_1Multf.nn.gelu_1/Gelu/mul:z:0tf.nn.gelu_1/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_12/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu_1/Gelu/mul_1:z:0dense_12_74765dense_12_74767*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_74764ã
dropout_4/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_74775·
tf.__operators__.add_4/AddV2AddV2"dropout_4/PartitionedCall:output:06layer_normalization_3/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_4_74801layer_normalization_4_74803*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_74800¤
 dense_14/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_14_74837dense_14_74839*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_74836¤
 dense_13/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_13_74873dense_13_74875*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_74872ç
reshape_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892é
reshape_5/PartitionedCall_1PartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892
'tf.compat.v1.transpose_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_9/transpose	Transpose"reshape_5/PartitionedCall:output:00tf.compat.v1.transpose_9/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_2/ShapeShape&tf.compat.v1.transpose_9/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_2/strided_sliceStridedSlice#tf.compat.v1.shape_2/Shape:output:07tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_2/CastCast1tf.__operators__.getitem_2/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_8/transpose	Transpose$reshape_5/PartitionedCall_1:output:00tf.compat.v1.transpose_8/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¤
 dense_15/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_15_74936dense_15_74938*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_74935Á
tf.linalg.matmul_4/MatMulBatchMatMulV2&tf.compat.v1.transpose_8/transpose:y:0&tf.compat.v1.transpose_9/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_2/SqrtSqrttf.cast_2/Cast:y:0*
T0*
_output_shapes
: é
reshape_5/PartitionedCall_2PartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892
tf.math.truediv_2/truedivRealDiv"tf.linalg.matmul_4/MatMul:output:0tf.math.sqrt_2/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_2/SoftmaxSoftmaxtf.math.truediv_2/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
(tf.compat.v1.transpose_10/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ã
#tf.compat.v1.transpose_10/transpose	Transpose$reshape_5/PartitionedCall_2:output:01tf.compat.v1.transpose_10/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1°
tf.linalg.matmul_5/MatMulBatchMatMulV2!tf.nn.softmax_2/Softmax:softmax:0'tf.compat.v1.transpose_10/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
(tf.compat.v1.transpose_11/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
#tf.compat.v1.transpose_11/transpose	Transpose"tf.linalg.matmul_5/MatMul:output:01tf.compat.v1.transpose_11/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1á
reshape_6/PartitionedCallPartitionedCall'tf.compat.v1.transpose_11/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_6_layer_call_and_return_conditional_losses_74964
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0dense_16_74997dense_16_74999*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_74996ã
dropout_5/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_75007·
tf.__operators__.add_5/AddV2AddV2"dropout_5/PartitionedCall:output:06layer_normalization_4/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_5_75033layer_normalization_5_75035*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_75032¥
 dense_17/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0dense_17_75069dense_17_75071*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_75068\
tf.nn.gelu_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
tf.nn.gelu_2/Gelu/mulMul tf.nn.gelu_2/Gelu/mul/x:output:0)dense_17/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?©
tf.nn.gelu_2/Gelu/truedivRealDiv)dense_17/StatefulPartitionedCall:output:0!tf.nn.gelu_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_2/Gelu/ErfErftf.nn.gelu_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_2/Gelu/addAddV2 tf.nn.gelu_2/Gelu/add/x:output:0tf.nn.gelu_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_2/Gelu/mul_1Multf.nn.gelu_2/Gelu/mul:z:0tf.nn.gelu_2/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu_2/Gelu/mul_1:z:0dense_18_75113dense_18_75115*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_75112ã
dropout_6/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_75123·
tf.__operators__.add_6/AddV2AddV2"dropout_6/PartitionedCall:output:06layer_normalization_5/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ó
flatten/PartitionedCallPartitionedCall tf.__operators__.add_6/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75132þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_75145dense_75147*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75144Ø
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_75155o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ñ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :($
"
_output_shapes
:1
¸
E
)__inference_reshape_2_layer_call_fn_79009

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_74268d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_74775

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¦
C
'__inference_flatten_layer_call_fn_80003

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75132a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

`
D__inference_reshape_4_layer_call_and_return_conditional_losses_74616

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ñ

'__inference_dense_2_layer_call_fn_78916

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_74140s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_3_layer_call_and_return_conditional_losses_79475

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ö

(__inference_dense_18_layer_call_fn_79941

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_75112s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

`
D__inference_reshape_6_layer_call_and_return_conditional_losses_74964

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ñ

'__inference_dense_1_layer_call_fn_78877

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_74176s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_79506

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ü
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_78965

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ó

(__inference_dense_15_layer_call_fn_79748

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_74935s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ãé
­(
@__inference_model_layer_call_and_return_conditional_losses_78681

inputs
tf_math_multiply_mul_y 
tf___operators___add_addv2_yG
9layer_normalization_batchnorm_mul_readvariableop_resource:C
5layer_normalization_batchnorm_readvariableop_resource:;
)dense_2_tensordot_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:;
)dense_1_tensordot_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:;
)dense_3_tensordot_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:;
)dense_4_tensordot_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:E
7layer_normalization_1_batchnorm_readvariableop_resource:<
)dense_5_tensordot_readvariableop_resource:	6
'dense_5_biasadd_readvariableop_resource:	<
)dense_6_tensordot_readvariableop_resource:	5
'dense_6_biasadd_readvariableop_resource:I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:E
7layer_normalization_2_batchnorm_readvariableop_resource:;
)dense_8_tensordot_readvariableop_resource:5
'dense_8_biasadd_readvariableop_resource:;
)dense_7_tensordot_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:;
)dense_9_tensordot_readvariableop_resource:5
'dense_9_biasadd_readvariableop_resource:<
*dense_10_tensordot_readvariableop_resource:6
(dense_10_biasadd_readvariableop_resource:I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:E
7layer_normalization_3_batchnorm_readvariableop_resource:=
*dense_11_tensordot_readvariableop_resource:	7
(dense_11_biasadd_readvariableop_resource:	=
*dense_12_tensordot_readvariableop_resource:	6
(dense_12_biasadd_readvariableop_resource:I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:E
7layer_normalization_4_batchnorm_readvariableop_resource:<
*dense_14_tensordot_readvariableop_resource:6
(dense_14_biasadd_readvariableop_resource:<
*dense_13_tensordot_readvariableop_resource:6
(dense_13_biasadd_readvariableop_resource:<
*dense_15_tensordot_readvariableop_resource:6
(dense_15_biasadd_readvariableop_resource:<
*dense_16_tensordot_readvariableop_resource:6
(dense_16_biasadd_readvariableop_resource:I
;layer_normalization_5_batchnorm_mul_readvariableop_resource:E
7layer_normalization_5_batchnorm_readvariableop_resource:=
*dense_17_tensordot_readvariableop_resource:	7
(dense_17_biasadd_readvariableop_resource:	=
*dense_18_tensordot_readvariableop_resource:	6
(dense_18_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	
3
%dense_biasadd_readvariableop_resource:

identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢ dense_1/Tensordot/ReadVariableOp¢dense_10/BiasAdd/ReadVariableOp¢!dense_10/Tensordot/ReadVariableOp¢dense_11/BiasAdd/ReadVariableOp¢!dense_11/Tensordot/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢!dense_12/Tensordot/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢!dense_13/Tensordot/ReadVariableOp¢dense_14/BiasAdd/ReadVariableOp¢!dense_14/Tensordot/ReadVariableOp¢dense_15/BiasAdd/ReadVariableOp¢!dense_15/Tensordot/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢!dense_16/Tensordot/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢!dense_17/Tensordot/ReadVariableOp¢dense_18/BiasAdd/ReadVariableOp¢!dense_18/Tensordot/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢ dense_2/Tensordot/ReadVariableOp¢dense_3/BiasAdd/ReadVariableOp¢ dense_3/Tensordot/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢ dense_5/Tensordot/ReadVariableOp¢dense_6/BiasAdd/ReadVariableOp¢ dense_6/Tensordot/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢ dense_7/Tensordot/ReadVariableOp¢dense_8/BiasAdd/ReadVariableOp¢ dense_8/Tensordot/ReadVariableOp¢dense_9/BiasAdd/ReadVariableOp¢ dense_9/Tensordot/ReadVariableOp¢,layer_normalization/batchnorm/ReadVariableOp¢0layer_normalization/batchnorm/mul/ReadVariableOp¢.layer_normalization_1/batchnorm/ReadVariableOp¢2layer_normalization_1/batchnorm/mul/ReadVariableOp¢.layer_normalization_2/batchnorm/ReadVariableOp¢2layer_normalization_2/batchnorm/mul/ReadVariableOp¢.layer_normalization_3/batchnorm/ReadVariableOp¢2layer_normalization_3/batchnorm/mul/ReadVariableOp¢.layer_normalization_4/batchnorm/ReadVariableOp¢2layer_normalization_4/batchnorm/mul/ReadVariableOp¢.layer_normalization_5/batchnorm/ReadVariableOp¢2layer_normalization_5/batchnorm/mul/ReadVariableOpÕ
,tf.image.extract_patches/ExtractImagePatchesExtractImagePatchesinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksizes
*
paddingVALID*
rates
*
strides
s
reshape/ShapeShape6tf.image.extract_patches/ExtractImagePatches:patches:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿY
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :¯
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:¨
reshape/ReshapeReshape6tf.image.extract_patches/ExtractImagePatches:patches:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.math.multiply/MulMulreshape/Reshape:output:0tf_math_multiply_mul_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout/dropout/MulMultf.__operators__.add/AddV2:z:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
dropout/dropout/ShapeShapetf.__operators__.add/AddV2:z:0*
T0*
_output_shapes
: 
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Â
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1|
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ç
 layer_normalization/moments/meanMeandropout/dropout/Mul_1:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Æ
-layer_normalization/moments/SquaredDifferenceSquaredDifferencedropout/dropout/Mul_1:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:ç
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(h
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75½
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¦
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Á
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
#layer_normalization/batchnorm/mul_1Muldropout/dropout/Mul_1:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1²
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0½
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1²
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
dense_2/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ª
dense_2/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
dense_1/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ª
dense_1/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1W
reshape_1/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Û
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_1/ReshapeReshapedense_2/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Y
reshape_1/Shape_1Shapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_slice_1StridedSlicereshape_1/Shape_1:output:0(reshape_1/strided_slice_1/stack:output:0*reshape_1/strided_slice_1/stack_1:output:0*reshape_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_1/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_1/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_1/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_1/Reshape_1/shapePack"reshape_1/strided_slice_1:output:0$reshape_1/Reshape_1/shape/1:output:0$reshape_1/Reshape_1/shape/2:output:0$reshape_1/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_1/Reshape_1Reshapedense_1/BiasAdd:output:0"reshape_1/Reshape_1/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ·
"tf.compat.v1.transpose_1/transpose	Transposereshape_1/Reshape:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.compat.v1.shape/ShapeShape&tf.compat.v1.transpose_1/transpose:y:0*
T0*
_output_shapes
:
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
tf.cast/CastCast/tf.__operators__.getitem/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             µ
 tf.compat.v1.transpose/transpose	Transposereshape_1/Reshape_1:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       n
dense_3/Tensordot/ShapeShape'layer_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ª
dense_3/Tensordot/transpose	Transpose'layer_normalization/batchnorm/add_1:z:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1½
tf.linalg.matmul/MatMulBatchMatMulV2$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(L
tf.math.sqrt/SqrtSqrttf.cast/Cast:y:0*
T0*
_output_shapes
: Y
reshape_1/Shape_2Shapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_1/strided_slice_2StridedSlicereshape_1/Shape_2:output:0(reshape_1/strided_slice_2/stack:output:0*reshape_1/strided_slice_2/stack_1:output:0*reshape_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_1/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_1/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_1/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_1/Reshape_2/shapePack"reshape_1/strided_slice_2:output:0$reshape_1/Reshape_2/shape/1:output:0$reshape_1/Reshape_2/shape/2:output:0$reshape_1/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_1/Reshape_2Reshapedense_3/BiasAdd:output:0"reshape_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.math.truediv/truedivRealDiv tf.linalg.matmul/MatMul:output:0tf.math.sqrt/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11w
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¹
"tf.compat.v1.transpose_2/transpose	Transposereshape_1/Reshape_2:output:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
tf.linalg.matmul_1/MatMulBatchMatMulV2tf.nn.softmax/Softmax:softmax:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_3/transpose	Transpose"tf.linalg.matmul_1/MatMul:output:00tf.compat.v1.transpose_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1e
reshape_2/ShapeShape&tf.compat.v1.transpose_3/transpose:y:0*
T0*
_output_shapes
:g
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :·
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_2/ReshapeReshape&tf.compat.v1.transpose_3/transpose:y:0 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       a
dense_4/Tensordot/ShapeShapereshape_2/Reshape:output:0*
T0*
_output_shapes
:a
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_4/Tensordot/transpose	Transposereshape_2/Reshape:output:0!dense_4/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_1/dropout/MulMuldense_4/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_
dropout_1/dropout/ShapeShapedense_4/BiasAdd:output:0*
T0*
_output_shapes
:¤
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=È
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¡
tf.__operators__.add_1/AddV2AddV2dropout_1/dropout/Mul_1:z:0'layer_normalization/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ò
"layer_normalization_1/moments/meanMean tf.__operators__.add_1/AddV2:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ñ
/layer_normalization_1/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_1/AddV2:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(j
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ç
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
%layer_normalization_1/batchnorm/mul_1Mul tf.__operators__.add_1/AddV2:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ã
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0`
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_5/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
dense_5/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
tf.nn.gelu/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
tf.nn.gelu/Gelu/mulMultf.nn.gelu/Gelu/mul/x:output:0dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1[
tf.nn.gelu/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
tf.nn.gelu/Gelu/truedivRealDivdense_5/BiasAdd:output:0tf.nn.gelu/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.nn.gelu/Gelu/ErfErftf.nn.gelu/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
tf.nn.gelu/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu/Gelu/addAddV2tf.nn.gelu/Gelu/add/x:output:0tf.nn.gelu/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu/Gelu/mul_1Multf.nn.gelu/Gelu/mul:z:0tf.nn.gelu/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0`
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       `
dense_6/Tensordot/ShapeShapetf.nn.gelu/Gelu/mul_1:z:0*
T0*
_output_shapes
:a
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_6/Tensordot/transpose	Transposetf.nn.gelu/Gelu/mul_1:z:0!dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_2/dropout/MulMuldense_6/BiasAdd:output:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_
dropout_2/dropout/ShapeShapedense_6/BiasAdd:output:0*
T0*
_output_shapes
:¤
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=È
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
tf.__operators__.add_2/AddV2AddV2dropout_2/dropout/Mul_1:z:0)layer_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ò
"layer_normalization_2/moments/meanMean tf.__operators__.add_2/AddV2:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ñ
/layer_normalization_2/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_2/AddV2:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(j
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ç
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
%layer_normalization_2/batchnorm/mul_1Mul tf.__operators__.add_2/AddV2:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ã
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_8/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
dense_8/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0!dense_8/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_7/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
dense_7/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1W
reshape_3/ShapeShapedense_8/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Û
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_3/ReshapeReshapedense_8/BiasAdd:output:0 reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Y
reshape_3/Shape_1Shapedense_7/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_3/strided_slice_1StridedSlicereshape_3/Shape_1:output:0(reshape_3/strided_slice_1/stack:output:0*reshape_3/strided_slice_1/stack_1:output:0*reshape_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_3/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_3/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_3/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_3/Reshape_1/shapePack"reshape_3/strided_slice_1:output:0$reshape_3/Reshape_1/shape/1:output:0$reshape_3/Reshape_1/shape/2:output:0$reshape_3/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_3/Reshape_1Reshapedense_7/BiasAdd:output:0"reshape_3/Reshape_1/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_5/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ·
"tf.compat.v1.transpose_5/transpose	Transposereshape_3/Reshape:output:00tf.compat.v1.transpose_5/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_1/ShapeShape&tf.compat.v1.transpose_5/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_1/CastCast1tf.__operators__.getitem_1/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¹
"tf.compat.v1.transpose_4/transpose	Transposereshape_3/Reshape_1:output:00tf.compat.v1.transpose_4/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
dense_9/Tensordot/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:a
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Û
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ¼
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¬
dense_9/Tensordot/transpose	Transpose)layer_normalization_2/batchnorm/add_1:z:0!dense_9/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ç
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Á
tf.linalg.matmul_2/MatMulBatchMatMulV2&tf.compat.v1.transpose_4/transpose:y:0&tf.compat.v1.transpose_5/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_1/SqrtSqrttf.cast_1/Cast:y:0*
T0*
_output_shapes
: Y
reshape_3/Shape_2Shapedense_9/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_3/strided_slice_2StridedSlicereshape_3/Shape_2:output:0(reshape_3/strided_slice_2/stack:output:0*reshape_3/strided_slice_2/stack_1:output:0*reshape_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_3/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_3/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_3/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_3/Reshape_2/shapePack"reshape_3/strided_slice_2:output:0$reshape_3/Reshape_2/shape/1:output:0$reshape_3/Reshape_2/shape/2:output:0$reshape_3/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_3/Reshape_2Reshapedense_9/BiasAdd:output:0"reshape_3/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.math.truediv_1/truedivRealDiv"tf.linalg.matmul_2/MatMul:output:0tf.math.sqrt_1/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_6/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¹
"tf.compat.v1.transpose_6/transpose	Transposereshape_3/Reshape_2:output:00tf.compat.v1.transpose_6/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¯
tf.linalg.matmul_3/MatMulBatchMatMulV2!tf.nn.softmax_1/Softmax:softmax:0&tf.compat.v1.transpose_6/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_7/transpose	Transpose"tf.linalg.matmul_3/MatMul:output:00tf.compat.v1.transpose_7/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1e
reshape_4/ShapeShape&tf.compat.v1.transpose_7/transpose:y:0*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :·
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_4/ReshapeReshape&tf.compat.v1.transpose_7/transpose:y:0 reshape_4/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_10/Tensordot/ShapeShapereshape_4/Reshape:output:0*
T0*
_output_shapes
:b
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_10/Tensordot/transpose	Transposereshape_4/Reshape:output:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_3/dropout/MulMuldense_10/BiasAdd:output:0 dropout_3/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
dropout_3/dropout/ShapeShapedense_10/BiasAdd:output:0*
T0*
_output_shapes
:¤
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=È
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
tf.__operators__.add_3/AddV2AddV2dropout_3/dropout/Mul_1:z:0)layer_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ò
"layer_normalization_3/moments/meanMean tf.__operators__.add_3/AddV2:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ñ
/layer_normalization_3/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_3/AddV2:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(j
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ç
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
%layer_normalization_3/batchnorm/mul_1Mul tf.__operators__.add_3/AddV2:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ã
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0a
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
dense_11/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:b
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
dense_11/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0"dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
tf.nn.gelu_1/Gelu/mulMul tf.nn.gelu_1/Gelu/mul/x:output:0dense_11/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
tf.nn.gelu_1/Gelu/truedivRealDivdense_11/BiasAdd:output:0!tf.nn.gelu_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_1/Gelu/ErfErftf.nn.gelu_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_1/Gelu/addAddV2 tf.nn.gelu_1/Gelu/add/x:output:0tf.nn.gelu_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_1/Gelu/mul_1Multf.nn.gelu_1/Gelu/mul:z:0tf.nn.gelu_1/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_12/Tensordot/ReadVariableOpReadVariableOp*dense_12_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0a
dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_12/Tensordot/ShapeShapetf.nn.gelu_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:b
 dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_12/Tensordot/GatherV2GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/free:output:0)dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_12/Tensordot/GatherV2_1GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/axes:output:0+dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_12/Tensordot/ProdProd$dense_12/Tensordot/GatherV2:output:0!dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_12/Tensordot/Prod_1Prod&dense_12/Tensordot/GatherV2_1:output:0#dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_12/Tensordot/concatConcatV2 dense_12/Tensordot/free:output:0 dense_12/Tensordot/axes:output:0'dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_12/Tensordot/stackPack dense_12/Tensordot/Prod:output:0"dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¡
dense_12/Tensordot/transpose	Transposetf.nn.gelu_1/Gelu/mul_1:z:0"dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_12/Tensordot/ReshapeReshape dense_12/Tensordot/transpose:y:0!dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_12/Tensordot/MatMulMatMul#dense_12/Tensordot/Reshape:output:0)dense_12/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_12/Tensordot/concat_1ConcatV2$dense_12/Tensordot/GatherV2:output:0#dense_12/Tensordot/Const_2:output:0)dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_12/TensordotReshape#dense_12/Tensordot/MatMul:product:0$dense_12/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_12/BiasAddBiasAdddense_12/Tensordot:output:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_4/dropout/MulMuldense_12/BiasAdd:output:0 dropout_4/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
dropout_4/dropout/ShapeShapedense_12/BiasAdd:output:0*
T0*
_output_shapes
:¤
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=È
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
tf.__operators__.add_4/AddV2AddV2dropout_4/dropout/Mul_1:z:0)layer_normalization_3/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ò
"layer_normalization_4/moments/meanMean tf.__operators__.add_4/AddV2:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ñ
/layer_normalization_4/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_4/AddV2:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(j
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ç
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
%layer_normalization_4/batchnorm/mul_1Mul tf.__operators__.add_4/AddV2:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ã
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_14/Tensordot/ReadVariableOpReadVariableOp*dense_14_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_14/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_14/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
dense_14/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:b
 dense_14/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_14/Tensordot/GatherV2GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/free:output:0)dense_14/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_14/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_14/Tensordot/GatherV2_1GatherV2!dense_14/Tensordot/Shape:output:0 dense_14/Tensordot/axes:output:0+dense_14/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_14/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/ProdProd$dense_14/Tensordot/GatherV2:output:0!dense_14/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_14/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_14/Tensordot/Prod_1Prod&dense_14/Tensordot/GatherV2_1:output:0#dense_14/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_14/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_14/Tensordot/concatConcatV2 dense_14/Tensordot/free:output:0 dense_14/Tensordot/axes:output:0'dense_14/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/Tensordot/stackPack dense_14/Tensordot/Prod:output:0"dense_14/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
dense_14/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0"dense_14/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_14/Tensordot/ReshapeReshape dense_14/Tensordot/transpose:y:0!dense_14/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_14/Tensordot/MatMulMatMul#dense_14/Tensordot/Reshape:output:0)dense_14/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_14/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_14/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_14/Tensordot/concat_1ConcatV2$dense_14/Tensordot/GatherV2:output:0#dense_14/Tensordot/Const_2:output:0)dense_14/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_14/TensordotReshape#dense_14/Tensordot/MatMul:product:0$dense_14/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_14/BiasAdd/ReadVariableOpReadVariableOp(dense_14_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_14/BiasAddBiasAdddense_14/Tensordot:output:0'dense_14/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_13/Tensordot/ReadVariableOpReadVariableOp*dense_13_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
dense_13/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:b
 dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_13/Tensordot/GatherV2GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/free:output:0)dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_13/Tensordot/GatherV2_1GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/axes:output:0+dense_13/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_13/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_13/Tensordot/ProdProd$dense_13/Tensordot/GatherV2:output:0!dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_13/Tensordot/Prod_1Prod&dense_13/Tensordot/GatherV2_1:output:0#dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_13/Tensordot/concatConcatV2 dense_13/Tensordot/free:output:0 dense_13/Tensordot/axes:output:0'dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_13/Tensordot/stackPack dense_13/Tensordot/Prod:output:0"dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
dense_13/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0"dense_13/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_13/Tensordot/ReshapeReshape dense_13/Tensordot/transpose:y:0!dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_13/Tensordot/MatMulMatMul#dense_13/Tensordot/Reshape:output:0)dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_13/Tensordot/concat_1ConcatV2$dense_13/Tensordot/GatherV2:output:0#dense_13/Tensordot/Const_2:output:0)dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_13/TensordotReshape#dense_13/Tensordot/MatMul:product:0$dense_13/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_13/BiasAddBiasAdddense_13/Tensordot:output:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1X
reshape_5/ShapeShapedense_14/BiasAdd:output:0*
T0*
_output_shapes
:g
reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_5/strided_sliceStridedSlicereshape_5/Shape:output:0&reshape_5/strided_slice/stack:output:0(reshape_5/strided_slice/stack_1:output:0(reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Û
reshape_5/Reshape/shapePack reshape_5/strided_slice:output:0"reshape_5/Reshape/shape/1:output:0"reshape_5/Reshape/shape/2:output:0"reshape_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_5/ReshapeReshapedense_14/BiasAdd:output:0 reshape_5/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
reshape_5/Shape_1Shapedense_13/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_5/strided_slice_1StridedSlicereshape_5/Shape_1:output:0(reshape_5/strided_slice_1/stack:output:0*reshape_5/strided_slice_1/stack_1:output:0*reshape_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_5/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_5/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_5/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_5/Reshape_1/shapePack"reshape_5/strided_slice_1:output:0$reshape_5/Reshape_1/shape/1:output:0$reshape_5/Reshape_1/shape/2:output:0$reshape_5/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_5/Reshape_1Reshapedense_13/BiasAdd:output:0"reshape_5/Reshape_1/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ·
"tf.compat.v1.transpose_9/transpose	Transposereshape_5/Reshape:output:00tf.compat.v1.transpose_9/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_2/ShapeShape&tf.compat.v1.transpose_9/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_2/strided_sliceStridedSlice#tf.compat.v1.shape_2/Shape:output:07tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_2/CastCast1tf.__operators__.getitem_2/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¹
"tf.compat.v1.transpose_8/transpose	Transposereshape_5/Reshape_1:output:00tf.compat.v1.transpose_8/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_15/Tensordot/ReadVariableOpReadVariableOp*dense_15_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_15/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_15/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
dense_15/Tensordot/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:b
 dense_15/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_15/Tensordot/GatherV2GatherV2!dense_15/Tensordot/Shape:output:0 dense_15/Tensordot/free:output:0)dense_15/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_15/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_15/Tensordot/GatherV2_1GatherV2!dense_15/Tensordot/Shape:output:0 dense_15/Tensordot/axes:output:0+dense_15/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_15/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_15/Tensordot/ProdProd$dense_15/Tensordot/GatherV2:output:0!dense_15/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_15/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_15/Tensordot/Prod_1Prod&dense_15/Tensordot/GatherV2_1:output:0#dense_15/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_15/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_15/Tensordot/concatConcatV2 dense_15/Tensordot/free:output:0 dense_15/Tensordot/axes:output:0'dense_15/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_15/Tensordot/stackPack dense_15/Tensordot/Prod:output:0"dense_15/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
dense_15/Tensordot/transpose	Transpose)layer_normalization_4/batchnorm/add_1:z:0"dense_15/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_15/Tensordot/ReshapeReshape dense_15/Tensordot/transpose:y:0!dense_15/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_15/Tensordot/MatMulMatMul#dense_15/Tensordot/Reshape:output:0)dense_15/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_15/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_15/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_15/Tensordot/concat_1ConcatV2$dense_15/Tensordot/GatherV2:output:0#dense_15/Tensordot/Const_2:output:0)dense_15/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_15/TensordotReshape#dense_15/Tensordot/MatMul:product:0$dense_15/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_15/BiasAddBiasAdddense_15/Tensordot:output:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Á
tf.linalg.matmul_4/MatMulBatchMatMulV2&tf.compat.v1.transpose_8/transpose:y:0&tf.compat.v1.transpose_9/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_2/SqrtSqrttf.cast_2/Cast:y:0*
T0*
_output_shapes
: Z
reshape_5/Shape_2Shapedense_15/BiasAdd:output:0*
T0*
_output_shapes
:i
reshape_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!reshape_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!reshape_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_5/strided_slice_2StridedSlicereshape_5/Shape_2:output:0(reshape_5/strided_slice_2/stack:output:0*reshape_5/strided_slice_2/stack_1:output:0*reshape_5/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
reshape_5/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ]
reshape_5/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :]
reshape_5/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :å
reshape_5/Reshape_2/shapePack"reshape_5/strided_slice_2:output:0$reshape_5/Reshape_2/shape/1:output:0$reshape_5/Reshape_2/shape/2:output:0$reshape_5/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_5/Reshape_2Reshapedense_15/BiasAdd:output:0"reshape_5/Reshape_2/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.math.truediv_2/truedivRealDiv"tf.linalg.matmul_4/MatMul:output:0tf.math.sqrt_2/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_2/SoftmaxSoftmaxtf.math.truediv_2/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
(tf.compat.v1.transpose_10/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             »
#tf.compat.v1.transpose_10/transpose	Transposereshape_5/Reshape_2:output:01tf.compat.v1.transpose_10/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1°
tf.linalg.matmul_5/MatMulBatchMatMulV2!tf.nn.softmax_2/Softmax:softmax:0'tf.compat.v1.transpose_10/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
(tf.compat.v1.transpose_11/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
#tf.compat.v1.transpose_11/transpose	Transpose"tf.linalg.matmul_5/MatMul:output:01tf.compat.v1.transpose_11/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
reshape_6/ShapeShape'tf.compat.v1.transpose_11/transpose:y:0*
T0*
_output_shapes
:g
reshape_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_6/strided_sliceStridedSlicereshape_6/Shape:output:0&reshape_6/strided_slice/stack:output:0(reshape_6/strided_slice/stack_1:output:0(reshape_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
reshape_6/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
reshape_6/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :·
reshape_6/Reshape/shapePack reshape_6/strided_slice:output:0"reshape_6/Reshape/shape/1:output:0"reshape_6/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape_6/ReshapeReshape'tf.compat.v1.transpose_11/transpose:y:0 reshape_6/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_16/Tensordot/ReadVariableOpReadVariableOp*dense_16_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_16/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_16/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       b
dense_16/Tensordot/ShapeShapereshape_6/Reshape:output:0*
T0*
_output_shapes
:b
 dense_16/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_16/Tensordot/GatherV2GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/free:output:0)dense_16/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_16/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_16/Tensordot/GatherV2_1GatherV2!dense_16/Tensordot/Shape:output:0 dense_16/Tensordot/axes:output:0+dense_16/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_16/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_16/Tensordot/ProdProd$dense_16/Tensordot/GatherV2:output:0!dense_16/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_16/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_16/Tensordot/Prod_1Prod&dense_16/Tensordot/GatherV2_1:output:0#dense_16/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_16/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_16/Tensordot/concatConcatV2 dense_16/Tensordot/free:output:0 dense_16/Tensordot/axes:output:0'dense_16/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_16/Tensordot/stackPack dense_16/Tensordot/Prod:output:0"dense_16/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_16/Tensordot/transpose	Transposereshape_6/Reshape:output:0"dense_16/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_16/Tensordot/ReshapeReshape dense_16/Tensordot/transpose:y:0!dense_16/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_16/Tensordot/MatMulMatMul#dense_16/Tensordot/Reshape:output:0)dense_16/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_16/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_16/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_16/Tensordot/concat_1ConcatV2$dense_16/Tensordot/GatherV2:output:0#dense_16/Tensordot/Const_2:output:0)dense_16/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_16/TensordotReshape#dense_16/Tensordot/MatMul:product:0$dense_16/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_16/BiasAddBiasAdddense_16/Tensordot:output:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_5/dropout/MulMuldense_16/BiasAdd:output:0 dropout_5/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
dropout_5/dropout/ShapeShapedense_16/BiasAdd:output:0*
T0*
_output_shapes
:¤
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=È
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
tf.__operators__.add_5/AddV2AddV2dropout_5/dropout/Mul_1:z:0)layer_normalization_4/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
4layer_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Ò
"layer_normalization_5/moments/meanMean tf.__operators__.add_5/AddV2:z:0=layer_normalization_5/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(
*layer_normalization_5/moments/StopGradientStopGradient+layer_normalization_5/moments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ñ
/layer_normalization_5/moments/SquaredDifferenceSquaredDifference tf.__operators__.add_5/AddV2:z:03layer_normalization_5/moments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
8layer_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:í
&layer_normalization_5/moments/varianceMean3layer_normalization_5/moments/SquaredDifference:z:0Alayer_normalization_5/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(j
%layer_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75Ã
#layer_normalization_5/batchnorm/addAddV2/layer_normalization_5/moments/variance:output:0.layer_normalization_5/batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%layer_normalization_5/batchnorm/RsqrtRsqrt'layer_normalization_5/batchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2layer_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0Ç
#layer_normalization_5/batchnorm/mulMul)layer_normalization_5/batchnorm/Rsqrt:y:0:layer_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
%layer_normalization_5/batchnorm/mul_1Mul tf.__operators__.add_5/AddV2:z:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_5/batchnorm/mul_2Mul+layer_normalization_5/moments/mean:output:0'layer_normalization_5/batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¢
.layer_normalization_5/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0Ã
#layer_normalization_5/batchnorm/subSub6layer_normalization_5/batchnorm/ReadVariableOp:value:0)layer_normalization_5/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¸
%layer_normalization_5/batchnorm/add_1AddV2)layer_normalization_5/batchnorm/mul_1:z:0'layer_normalization_5/batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_17/Tensordot/ReadVariableOpReadVariableOp*dense_17_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0a
dense_17/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_17/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
dense_17/Tensordot/ShapeShape)layer_normalization_5/batchnorm/add_1:z:0*
T0*
_output_shapes
:b
 dense_17/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_17/Tensordot/GatherV2GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/free:output:0)dense_17/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_17/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_17/Tensordot/GatherV2_1GatherV2!dense_17/Tensordot/Shape:output:0 dense_17/Tensordot/axes:output:0+dense_17/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_17/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_17/Tensordot/ProdProd$dense_17/Tensordot/GatherV2:output:0!dense_17/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_17/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_17/Tensordot/Prod_1Prod&dense_17/Tensordot/GatherV2_1:output:0#dense_17/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_17/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_17/Tensordot/concatConcatV2 dense_17/Tensordot/free:output:0 dense_17/Tensordot/axes:output:0'dense_17/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_17/Tensordot/stackPack dense_17/Tensordot/Prod:output:0"dense_17/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:®
dense_17/Tensordot/transpose	Transpose)layer_normalization_5/batchnorm/add_1:z:0"dense_17/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_17/Tensordot/ReshapeReshape dense_17/Tensordot/transpose:y:0!dense_17/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¦
dense_17/Tensordot/MatMulMatMul#dense_17/Tensordot/Reshape:output:0)dense_17/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
dense_17/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_17/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_17/Tensordot/concat_1ConcatV2$dense_17/Tensordot/GatherV2:output:0#dense_17/Tensordot/Const_2:output:0)dense_17/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_17/TensordotReshape#dense_17/Tensordot/MatMul:product:0$dense_17/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_17/BiasAddBiasAdddense_17/Tensordot:output:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
tf.nn.gelu_2/Gelu/mulMul tf.nn.gelu_2/Gelu/mul/x:output:0dense_17/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?
tf.nn.gelu_2/Gelu/truedivRealDivdense_17/BiasAdd:output:0!tf.nn.gelu_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_2/Gelu/ErfErftf.nn.gelu_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_2/Gelu/addAddV2 tf.nn.gelu_2/Gelu/add/x:output:0tf.nn.gelu_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_2/Gelu/mul_1Multf.nn.gelu_2/Gelu/mul:z:0tf.nn.gelu_2/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!dense_18/Tensordot/ReadVariableOpReadVariableOp*dense_18_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0a
dense_18/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_18/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_18/Tensordot/ShapeShapetf.nn.gelu_2/Gelu/mul_1:z:0*
T0*
_output_shapes
:b
 dense_18/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_18/Tensordot/GatherV2GatherV2!dense_18/Tensordot/Shape:output:0 dense_18/Tensordot/free:output:0)dense_18/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_18/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_18/Tensordot/GatherV2_1GatherV2!dense_18/Tensordot/Shape:output:0 dense_18/Tensordot/axes:output:0+dense_18/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_18/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_18/Tensordot/ProdProd$dense_18/Tensordot/GatherV2:output:0!dense_18/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_18/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_18/Tensordot/Prod_1Prod&dense_18/Tensordot/GatherV2_1:output:0#dense_18/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_18/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_18/Tensordot/concatConcatV2 dense_18/Tensordot/free:output:0 dense_18/Tensordot/axes:output:0'dense_18/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_18/Tensordot/stackPack dense_18/Tensordot/Prod:output:0"dense_18/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¡
dense_18/Tensordot/transpose	Transposetf.nn.gelu_2/Gelu/mul_1:z:0"dense_18/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¥
dense_18/Tensordot/ReshapeReshape dense_18/Tensordot/transpose:y:0!dense_18/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_18/Tensordot/MatMulMatMul#dense_18/Tensordot/Reshape:output:0)dense_18/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_18/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_18/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_18/Tensordot/concat_1ConcatV2$dense_18/Tensordot/GatherV2:output:0#dense_18/Tensordot/Const_2:output:0)dense_18/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_18/TensordotReshape#dense_18/Tensordot/MatMul:product:0$dense_18/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_18/BiasAddBiasAdddense_18/Tensordot:output:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
dropout_6/dropout/MulMuldense_18/BiasAdd:output:0 dropout_6/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
dropout_6/dropout/ShapeShapedense_18/BiasAdd:output:0*
T0*
_output_shapes
:¤
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=È
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1£
tf.__operators__.add_6/AddV2AddV2dropout_6/dropout/Mul_1:z:0)layer_normalization_5/batchnorm/add_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
flatten/ReshapeReshape tf.__operators__.add_6/AddV2:z:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d
softmax/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
h
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
½
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp"^dense_12/Tensordot/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp"^dense_13/Tensordot/ReadVariableOp ^dense_14/BiasAdd/ReadVariableOp"^dense_14/Tensordot/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp"^dense_15/Tensordot/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp"^dense_16/Tensordot/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp"^dense_17/Tensordot/ReadVariableOp ^dense_18/BiasAdd/ReadVariableOp"^dense_18/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp/^layer_normalization_5/batchnorm/ReadVariableOp3^layer_normalization_5/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2F
!dense_12/Tensordot/ReadVariableOp!dense_12/Tensordot/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2F
!dense_13/Tensordot/ReadVariableOp!dense_13/Tensordot/ReadVariableOp2B
dense_14/BiasAdd/ReadVariableOpdense_14/BiasAdd/ReadVariableOp2F
!dense_14/Tensordot/ReadVariableOp!dense_14/Tensordot/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2F
!dense_15/Tensordot/ReadVariableOp!dense_15/Tensordot/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2F
!dense_16/Tensordot/ReadVariableOp!dense_16/Tensordot/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2F
!dense_17/Tensordot/ReadVariableOp!dense_17/Tensordot/ReadVariableOp2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2F
!dense_18/Tensordot/ReadVariableOp!dense_18/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2`
.layer_normalization_5/batchnorm/ReadVariableOp.layer_normalization_5/batchnorm/ReadVariableOp2h
2layer_normalization_5/batchnorm/mul/ReadVariableOp2layer_normalization_5/batchnorm/mul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :($
"
_output_shapes
:1
ï

`
D__inference_reshape_2_layer_call_and_return_conditional_losses_79022

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_4_layer_call_and_return_conditional_losses_74300

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ê
ú
C__inference_dense_15_layer_call_and_return_conditional_losses_79778

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
b
D__inference_dropout_4_layer_call_and_return_conditional_losses_79599

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

Ü
%__inference_model_layer_call_fn_75265
input_1
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:	

unknown_46:	

unknown_47:	

unknown_48:

unknown_49:	


unknown_50:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_75158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: :($
"
_output_shapes
:1
ç
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_74659

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_5_layer_call_and_return_conditional_losses_75360

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_79463

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ó
ü
C__inference_dense_11_layer_call_and_return_conditional_losses_74720

inputs4
!tensordot_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ê
ú
C__inference_dense_16_layer_call_and_return_conditional_losses_74996

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
é

5__inference_layer_normalization_5_layer_call_fn_79871

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_75032s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
°
E
)__inference_dropout_4_layer_call_fn_79589

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_74775d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ï
 
!__inference__traced_restore_80397
file_prefix8
*assignvariableop_layer_normalization_gamma:9
+assignvariableop_1_layer_normalization_beta:3
!assignvariableop_2_dense_1_kernel:-
assignvariableop_3_dense_1_bias:3
!assignvariableop_4_dense_2_kernel:-
assignvariableop_5_dense_2_bias:3
!assignvariableop_6_dense_3_kernel:-
assignvariableop_7_dense_3_bias:3
!assignvariableop_8_dense_4_kernel:-
assignvariableop_9_dense_4_bias:=
/assignvariableop_10_layer_normalization_1_gamma:<
.assignvariableop_11_layer_normalization_1_beta:5
"assignvariableop_12_dense_5_kernel:	/
 assignvariableop_13_dense_5_bias:	5
"assignvariableop_14_dense_6_kernel:	.
 assignvariableop_15_dense_6_bias:=
/assignvariableop_16_layer_normalization_2_gamma:<
.assignvariableop_17_layer_normalization_2_beta:4
"assignvariableop_18_dense_7_kernel:.
 assignvariableop_19_dense_7_bias:4
"assignvariableop_20_dense_8_kernel:.
 assignvariableop_21_dense_8_bias:4
"assignvariableop_22_dense_9_kernel:.
 assignvariableop_23_dense_9_bias:5
#assignvariableop_24_dense_10_kernel:/
!assignvariableop_25_dense_10_bias:=
/assignvariableop_26_layer_normalization_3_gamma:<
.assignvariableop_27_layer_normalization_3_beta:6
#assignvariableop_28_dense_11_kernel:	0
!assignvariableop_29_dense_11_bias:	6
#assignvariableop_30_dense_12_kernel:	/
!assignvariableop_31_dense_12_bias:=
/assignvariableop_32_layer_normalization_4_gamma:<
.assignvariableop_33_layer_normalization_4_beta:5
#assignvariableop_34_dense_13_kernel:/
!assignvariableop_35_dense_13_bias:5
#assignvariableop_36_dense_14_kernel:/
!assignvariableop_37_dense_14_bias:5
#assignvariableop_38_dense_15_kernel:/
!assignvariableop_39_dense_15_bias:5
#assignvariableop_40_dense_16_kernel:/
!assignvariableop_41_dense_16_bias:=
/assignvariableop_42_layer_normalization_5_gamma:<
.assignvariableop_43_layer_normalization_5_beta:6
#assignvariableop_44_dense_17_kernel:	0
!assignvariableop_45_dense_17_bias:	6
#assignvariableop_46_dense_18_kernel:	/
!assignvariableop_47_dense_18_bias:3
 assignvariableop_48_dense_kernel:	
,
assignvariableop_49_dense_bias:
#
assignvariableop_50_total: #
assignvariableop_51_count: %
assignvariableop_52_total_1: %
assignvariableop_53_count_1: 
identity_55¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9»
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*á
value×BÔ7B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHß
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ´
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ò
_output_shapesß
Ü:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_10AssignVariableOp/assignvariableop_10_layer_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp.assignvariableop_11_layer_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_5_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_6_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_6_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_16AssignVariableOp/assignvariableop_16_layer_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp.assignvariableop_17_layer_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_7_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_7_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_8_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_8_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_9_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_9_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_10_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_10_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_26AssignVariableOp/assignvariableop_26_layer_normalization_3_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp.assignvariableop_27_layer_normalization_3_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_11_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp!assignvariableop_29_dense_11_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp#assignvariableop_30_dense_12_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp!assignvariableop_31_dense_12_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_32AssignVariableOp/assignvariableop_32_layer_normalization_4_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp.assignvariableop_33_layer_normalization_4_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp#assignvariableop_34_dense_13_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp!assignvariableop_35_dense_13_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_14_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_14_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp#assignvariableop_38_dense_15_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp!assignvariableop_39_dense_15_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp#assignvariableop_40_dense_16_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp!assignvariableop_41_dense_16_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_42AssignVariableOp/assignvariableop_42_layer_normalization_5_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp.assignvariableop_43_layer_normalization_5_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp#assignvariableop_44_dense_17_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp!assignvariableop_45_dense_17_biasIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp#assignvariableop_46_dense_18_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp!assignvariableop_47_dense_18_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp assignvariableop_48_dense_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOpassignvariableop_49_dense_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOpassignvariableop_50_totalIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOpassignvariableop_51_countIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOpassignvariableop_52_total_1Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOpassignvariableop_53_count_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ó	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: à	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_55Identity_55:output:0*
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ã
Ú
#__inference_signature_wrapper_78792
input_1
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:	

unknown_46:	

unknown_47:	

unknown_48:

unknown_49:	


unknown_50:

identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_74048o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: :($
"
_output_shapes
:1


c
D__inference_dropout_1_layer_call_and_return_conditional_losses_79088

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_74684

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
í

^
B__inference_reshape_layer_call_and_return_conditional_losses_78810

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
ù
B__inference_dense_9_layer_call_and_return_conditional_losses_79391

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ï
û
C__inference_dense_12_layer_call_and_return_conditional_losses_79584

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ï
û
C__inference_dense_18_layer_call_and_return_conditional_losses_79971

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ü
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_8_layer_call_and_return_conditional_losses_79333

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ç	
ò
@__inference_dense_layer_call_and_return_conditional_losses_80028

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×

(__inference_dense_11_layer_call_fn_79515

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_74720t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ï
û
C__inference_dense_12_layer_call_and_return_conditional_losses_74764

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_74336

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

b
)__inference_dropout_6_layer_call_fn_79981

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_75307s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ122
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Î
ú
B__inference_dense_6_layer_call_and_return_conditional_losses_74416

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

b
)__inference_dropout_3_layer_call_fn_79458

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_75498s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ122
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Õ

'__inference_dense_5_layer_call_fn_79128

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_74372t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_2_layer_call_and_return_conditional_losses_75583

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
å
`
B__inference_dropout_layer_call_and_return_conditional_losses_78825

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_79076

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Î
ú
B__inference_dense_6_layer_call_and_return_conditional_losses_79197

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ê
ú
C__inference_dense_13_layer_call_and_return_conditional_losses_79681

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
°
E
)__inference_dropout_6_layer_call_fn_79976

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_75123d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_74452

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
b
D__inference_dropout_6_layer_call_and_return_conditional_losses_79986

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ó

(__inference_dense_13_layer_call_fn_79651

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_74872s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
c
±
__inference__traced_save_80225
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop:
6savev2_layer_normalization_3_gamma_read_readvariableop9
5savev2_layer_normalization_3_beta_read_readvariableop.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop:
6savev2_layer_normalization_4_gamma_read_readvariableop9
5savev2_layer_normalization_4_beta_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop.
*savev2_dense_14_kernel_read_readvariableop,
(savev2_dense_14_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop:
6savev2_layer_normalization_5_gamma_read_readvariableop9
5savev2_layer_normalization_5_beta_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const_2

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¸
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*á
value×BÔ7B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-23/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-23/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-24/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-24/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÜ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ô
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop6savev2_layer_normalization_3_gamma_read_readvariableop5savev2_layer_normalization_3_beta_read_readvariableop*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop6savev2_layer_normalization_4_gamma_read_readvariableop5savev2_layer_normalization_4_beta_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop*savev2_dense_14_kernel_read_readvariableop(savev2_dense_14_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop6savev2_layer_normalization_5_gamma_read_readvariableop5savev2_layer_normalization_5_beta_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *E
dtypes;
927
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*£
_input_shapes
: :::::::::::::	::	::::::::::::::	::	::::::::::::::	::	::	
:
: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::%!

_output_shapes
:	:  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

:: *

_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
::%-!

_output_shapes
:	:!.

_output_shapes	
::%/!

_output_shapes
:	: 0

_output_shapes
::%1!

_output_shapes
:	
: 2

_output_shapes
:
:3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: 
¬
C
'__inference_dropout_layer_call_fn_78815

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_74080d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
í

^
B__inference_reshape_layer_call_and_return_conditional_losses_74069

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å

3__inference_layer_normalization_layer_call_fn_78846

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_74104s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

`
D__inference_reshape_4_layer_call_and_return_conditional_losses_79409

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ü
`
D__inference_reshape_5_layer_call_and_return_conditional_losses_79739

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ó
ü
C__inference_dense_17_layer_call_and_return_conditional_losses_75068

inputs4
!tensordot_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ë
^
B__inference_softmax_layer_call_and_return_conditional_losses_75155

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ï

P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_79642

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_75032

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

b
)__inference_dropout_2_layer_call_fn_79207

inputs
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_75583s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ122
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¸
E
)__inference_reshape_5_layer_call_fn_79725

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ò
û
B__inference_dense_5_layer_call_and_return_conditional_losses_79158

inputs4
!tensordot_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_75721

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

Ü
%__inference_model_layer_call_fn_76300
input_1
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:	

unknown_46:	

unknown_47:	

unknown_48:

unknown_49:	


unknown_50:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_76084o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: :($
"
_output_shapes
:1
Ò
û
B__inference_dense_5_layer_call_and_return_conditional_losses_74372

inputs4
!tensordot_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1d
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ö

(__inference_dense_12_layer_call_fn_79554

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_74764s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ó

(__inference_dense_10_layer_call_fn_79418

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_74648s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
°
E
)__inference_dropout_3_layer_call_fn_79453

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_74659d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_2_layer_call_and_return_conditional_losses_78946

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_4_layer_call_and_return_conditional_losses_79611

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_3_layer_call_and_return_conditional_losses_79004

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¼
^
B__inference_flatten_layer_call_and_return_conditional_losses_80009

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_9_layer_call_and_return_conditional_losses_74587

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
í

N__inference_layer_normalization_layer_call_and_return_conditional_losses_74104

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_1_layer_call_and_return_conditional_losses_75636

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_4_layer_call_and_return_conditional_losses_75445

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ñ

'__inference_dense_3_layer_call_fn_78974

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_74239s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ï

P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_79119

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

Û
%__inference_model_layer_call_fn_76895

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:	

unknown_46:	

unknown_47:	

unknown_48:

unknown_49:	


unknown_50:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_75158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :($
"
_output_shapes
:1
É
ù
B__inference_dense_7_layer_call_and_return_conditional_losses_79294

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ë
^
B__inference_softmax_layer_call_and_return_conditional_losses_80038

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¸
E
)__inference_reshape_6_layer_call_fn_79783

inputs
identity¶
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_6_layer_call_and_return_conditional_losses_74964d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¸
E
)__inference_reshape_3_layer_call_fn_79338

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ê
ú
C__inference_dense_13_layer_call_and_return_conditional_losses_74872

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ï
û
C__inference_dense_18_layer_call_and_return_conditional_losses_75112

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
þ
`
'__inference_dropout_layer_call_fn_78820

inputs
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_75721s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ122
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
é

5__inference_layer_normalization_3_layer_call_fn_79484

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_74684s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_6_layer_call_and_return_conditional_losses_79998

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
b
D__inference_dropout_5_layer_call_and_return_conditional_losses_79850

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_2_layer_call_and_return_conditional_losses_74140

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
åª
û
@__inference_model_layer_call_and_return_conditional_losses_76543
input_1
tf_math_multiply_mul_y 
tf___operators___add_addv2_y'
layer_normalization_76310:'
layer_normalization_76312:
dense_2_76315:
dense_2_76317:
dense_1_76320:
dense_1_76322:
dense_3_76337:
dense_3_76339:
dense_4_76353:
dense_4_76355:)
layer_normalization_1_76360:)
layer_normalization_1_76362: 
dense_5_76365:	
dense_5_76367:	 
dense_6_76378:	
dense_6_76380:)
layer_normalization_2_76385:)
layer_normalization_2_76387:
dense_8_76390:
dense_8_76392:
dense_7_76395:
dense_7_76397:
dense_9_76412:
dense_9_76414: 
dense_10_76428:
dense_10_76430:)
layer_normalization_3_76435:)
layer_normalization_3_76437:!
dense_11_76440:	
dense_11_76442:	!
dense_12_76453:	
dense_12_76455:)
layer_normalization_4_76460:)
layer_normalization_4_76462: 
dense_14_76465:
dense_14_76467: 
dense_13_76470:
dense_13_76472: 
dense_15_76487:
dense_15_76489: 
dense_16_76503:
dense_16_76505:)
layer_normalization_5_76510:)
layer_normalization_5_76512:!
dense_17_76515:	
dense_17_76517:	!
dense_18_76528:	
dense_18_76530:
dense_76536:	

dense_76538:

identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢ dense_10/StatefulPartitionedCall¢ dense_11/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢ dense_14/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢ dense_18/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCall¢dense_9/StatefulPartitionedCall¢+layer_normalization/StatefulPartitionedCall¢-layer_normalization_1/StatefulPartitionedCall¢-layer_normalization_2/StatefulPartitionedCall¢-layer_normalization_3/StatefulPartitionedCall¢-layer_normalization_4/StatefulPartitionedCall¢-layer_normalization_5/StatefulPartitionedCallÖ
,tf.image.extract_patches/ExtractImagePatchesExtractImagePatchesinput_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksizes
*
paddingVALID*
rates
*
strides
ì
reshape/PartitionedCallPartitionedCall6tf.image.extract_patches/ExtractImagePatches:patches:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_74069
tf.math.multiply/MulMul reshape/PartitionedCall:output:0tf_math_multiply_mul_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.__operators__.add/AddV2AddV2tf.math.multiply/Mul:z:0tf___operators___add_addv2_y*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ô
dropout/PartitionedCallPartitionedCalltf.__operators__.add/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_74080º
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0layer_normalization_76310layer_normalization_76312*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_layer_normalization_layer_call_and_return_conditional_losses_74104
dense_2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_2_76315dense_2_76317*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_74140
dense_1/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_1_76320dense_1_76322*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_74176æ
reshape_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196è
reshape_1/PartitionedCall_1PartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196
'tf.compat.v1.transpose_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_1/transpose	Transpose"reshape_1/PartitionedCall:output:00tf.compat.v1.transpose_1/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.compat.v1.shape/ShapeShape&tf.compat.v1.transpose_1/transpose:y:0*
T0*
_output_shapes
:
,tf.__operators__.getitem/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿx
.tf.__operators__.getitem/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: x
.tf.__operators__.getitem/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
&tf.__operators__.getitem/strided_sliceStridedSlice!tf.compat.v1.shape/Shape:output:05tf.__operators__.getitem/strided_slice/stack:output:07tf.__operators__.getitem/strided_slice/stack_1:output:07tf.__operators__.getitem/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
tf.cast/CastCast/tf.__operators__.getitem/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ~
%tf.compat.v1.transpose/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ½
 tf.compat.v1.transpose/transpose	Transpose$reshape_1/PartitionedCall_1:output:0.tf.compat.v1.transpose/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_3/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_3_76337dense_3_76339*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_74239½
tf.linalg.matmul/MatMulBatchMatMulV2$tf.compat.v1.transpose/transpose:y:0&tf.compat.v1.transpose_1/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(L
tf.math.sqrt/SqrtSqrttf.cast/Cast:y:0*
T0*
_output_shapes
: è
reshape_1/PartitionedCall_2PartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_74196
tf.math.truediv/truedivRealDiv tf.linalg.matmul/MatMul:output:0tf.math.sqrt/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11w
tf.nn.softmax/SoftmaxSoftmaxtf.math.truediv/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_2/transpose	Transpose$reshape_1/PartitionedCall_2:output:00tf.compat.v1.transpose_2/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
tf.linalg.matmul_1/MatMulBatchMatMulV2tf.nn.softmax/Softmax:softmax:0&tf.compat.v1.transpose_2/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_3/transpose	Transpose"tf.linalg.matmul_1/MatMul:output:00tf.compat.v1.transpose_3/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1à
reshape_2/PartitionedCallPartitionedCall&tf.compat.v1.transpose_3/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_74268
dense_4/StatefulPartitionedCallStatefulPartitionedCall"reshape_2/PartitionedCall:output:0dense_4_76353dense_4_76355*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_74300â
dropout_1/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_74311µ
tf.__operators__.add_1/AddV2AddV2"dropout_1/PartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_1/AddV2:z:0layer_normalization_1_76360layer_normalization_1_76362*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_74336¡
dense_5/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_5_76365dense_5_76367*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_74372Z
tf.nn.gelu/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
tf.nn.gelu/Gelu/mulMultf.nn.gelu/Gelu/mul/x:output:0(dense_5/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1[
tf.nn.gelu/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?¤
tf.nn.gelu/Gelu/truedivRealDiv(dense_5/StatefulPartitionedCall:output:0tf.nn.gelu/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1n
tf.nn.gelu/Gelu/ErfErftf.nn.gelu/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Z
tf.nn.gelu/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu/Gelu/addAddV2tf.nn.gelu/Gelu/add/x:output:0tf.nn.gelu/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu/Gelu/mul_1Multf.nn.gelu/Gelu/mul:z:0tf.nn.gelu/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dense_6/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu/Gelu/mul_1:z:0dense_6_76378dense_6_76380*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_74416â
dropout_2/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_74427·
tf.__operators__.add_2/AddV2AddV2"dropout_2/PartitionedCall:output:06layer_normalization_1/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_2/AddV2:z:0layer_normalization_2_76385layer_normalization_2_76387*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_74452 
dense_8/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_8_76390dense_8_76392*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_74488 
dense_7/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_7_76395dense_7_76397*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_74524æ
reshape_3/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544è
reshape_3/PartitionedCall_1PartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544
'tf.compat.v1.transpose_5/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_5/transpose	Transpose"reshape_3/PartitionedCall:output:00tf.compat.v1.transpose_5/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_1/ShapeShape&tf.compat.v1.transpose_5/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_1/strided_sliceStridedSlice#tf.compat.v1.shape_1/Shape:output:07tf.__operators__.getitem_1/strided_slice/stack:output:09tf.__operators__.getitem_1/strided_slice/stack_1:output:09tf.__operators__.getitem_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_1/CastCast1tf.__operators__.getitem_1/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_4/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_4/transpose	Transpose$reshape_3/PartitionedCall_1:output:00tf.compat.v1.transpose_4/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1 
dense_9/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0dense_9_76412dense_9_76414*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_74587Á
tf.linalg.matmul_2/MatMulBatchMatMulV2&tf.compat.v1.transpose_4/transpose:y:0&tf.compat.v1.transpose_5/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_1/SqrtSqrttf.cast_1/Cast:y:0*
T0*
_output_shapes
: è
reshape_3/PartitionedCall_2PartitionedCall(dense_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_74544
tf.math.truediv_1/truedivRealDiv"tf.linalg.matmul_2/MatMul:output:0tf.math.sqrt_1/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_1/SoftmaxSoftmaxtf.math.truediv_1/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
'tf.compat.v1.transpose_6/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_6/transpose	Transpose$reshape_3/PartitionedCall_2:output:00tf.compat.v1.transpose_6/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¯
tf.linalg.matmul_3/MatMulBatchMatMulV2!tf.nn.softmax_1/Softmax:softmax:0&tf.compat.v1.transpose_6/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
'tf.compat.v1.transpose_7/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_7/transpose	Transpose"tf.linalg.matmul_3/MatMul:output:00tf.compat.v1.transpose_7/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1à
reshape_4/PartitionedCallPartitionedCall&tf.compat.v1.transpose_7/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_74616
 dense_10/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0dense_10_76428dense_10_76430*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_74648ã
dropout_3/PartitionedCallPartitionedCall)dense_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_74659·
tf.__operators__.add_3/AddV2AddV2"dropout_3/PartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_3/AddV2:z:0layer_normalization_3_76435layer_normalization_3_76437*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_74684¥
 dense_11/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_11_76440dense_11_76442*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_11_layer_call_and_return_conditional_losses_74720\
tf.nn.gelu_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
tf.nn.gelu_1/Gelu/mulMul tf.nn.gelu_1/Gelu/mul/x:output:0)dense_11/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?©
tf.nn.gelu_1/Gelu/truedivRealDiv)dense_11/StatefulPartitionedCall:output:0!tf.nn.gelu_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_1/Gelu/ErfErftf.nn.gelu_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_1/Gelu/addAddV2 tf.nn.gelu_1/Gelu/add/x:output:0tf.nn.gelu_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_1/Gelu/mul_1Multf.nn.gelu_1/Gelu/mul:z:0tf.nn.gelu_1/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_12/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu_1/Gelu/mul_1:z:0dense_12_76453dense_12_76455*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_74764ã
dropout_4/PartitionedCallPartitionedCall)dense_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_4_layer_call_and_return_conditional_losses_74775·
tf.__operators__.add_4/AddV2AddV2"dropout_4/PartitionedCall:output:06layer_normalization_3/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_4/AddV2:z:0layer_normalization_4_76460layer_normalization_4_76462*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_74800¤
 dense_14/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_14_76465dense_14_76467*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_14_layer_call_and_return_conditional_losses_74836¤
 dense_13/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_13_76470dense_13_76472*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_74872ç
reshape_5/PartitionedCallPartitionedCall)dense_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892é
reshape_5/PartitionedCall_1PartitionedCall)dense_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892
'tf.compat.v1.transpose_9/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             ¿
"tf.compat.v1.transpose_9/transpose	Transpose"reshape_5/PartitionedCall:output:00tf.compat.v1.transpose_9/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1p
tf.compat.v1.shape_2/ShapeShape&tf.compat.v1.transpose_9/transpose:y:0*
T0*
_output_shapes
:
.tf.__operators__.getitem_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿz
0tf.__operators__.getitem_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: z
0tf.__operators__.getitem_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ò
(tf.__operators__.getitem_2/strided_sliceStridedSlice#tf.compat.v1.shape_2/Shape:output:07tf.__operators__.getitem_2/strided_slice/stack:output:09tf.__operators__.getitem_2/strided_slice/stack_1:output:09tf.__operators__.getitem_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
tf.cast_2/CastCast1tf.__operators__.getitem_2/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
'tf.compat.v1.transpose_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
"tf.compat.v1.transpose_8/transpose	Transpose$reshape_5/PartitionedCall_1:output:00tf.compat.v1.transpose_8/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1¤
 dense_15/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0dense_15_76487dense_15_76489*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_74935Á
tf.linalg.matmul_4/MatMulBatchMatMulV2&tf.compat.v1.transpose_8/transpose:y:0&tf.compat.v1.transpose_9/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11*
adj_y(P
tf.math.sqrt_2/SqrtSqrttf.cast_2/Cast:y:0*
T0*
_output_shapes
: é
reshape_5/PartitionedCall_2PartitionedCall)dense_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892
tf.math.truediv_2/truedivRealDiv"tf.linalg.matmul_4/MatMul:output:0tf.math.sqrt_2/Sqrt:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11{
tf.nn.softmax_2/SoftmaxSoftmaxtf.math.truediv_2/truediv:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ11
(tf.compat.v1.transpose_10/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Ã
#tf.compat.v1.transpose_10/transpose	Transpose$reshape_5/PartitionedCall_2:output:01tf.compat.v1.transpose_10/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1°
tf.linalg.matmul_5/MatMulBatchMatMulV2!tf.nn.softmax_2/Softmax:softmax:0'tf.compat.v1.transpose_10/transpose:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
(tf.compat.v1.transpose_11/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             Á
#tf.compat.v1.transpose_11/transpose	Transpose"tf.linalg.matmul_5/MatMul:output:01tf.compat.v1.transpose_11/transpose/perm:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1á
reshape_6/PartitionedCallPartitionedCall'tf.compat.v1.transpose_11/transpose:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_6_layer_call_and_return_conditional_losses_74964
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"reshape_6/PartitionedCall:output:0dense_16_76503dense_16_76505*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_74996ã
dropout_5/PartitionedCallPartitionedCall)dense_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_5_layer_call_and_return_conditional_losses_75007·
tf.__operators__.add_5/AddV2AddV2"dropout_5/PartitionedCall:output:06layer_normalization_4/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Â
-layer_normalization_5/StatefulPartitionedCallStatefulPartitionedCall tf.__operators__.add_5/AddV2:z:0layer_normalization_5_76510layer_normalization_5_76512*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_75032¥
 dense_17/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_5/StatefulPartitionedCall:output:0dense_17_76515dense_17_76517*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_75068\
tf.nn.gelu_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ? 
tf.nn.gelu_2/Gelu/mulMul tf.nn.gelu_2/Gelu/mul/x:output:0)dense_17/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
tf.nn.gelu_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *óµ?©
tf.nn.gelu_2/Gelu/truedivRealDiv)dense_17/StatefulPartitionedCall:output:0!tf.nn.gelu_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
tf.nn.gelu_2/Gelu/ErfErftf.nn.gelu_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1\
tf.nn.gelu_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tf.nn.gelu_2/Gelu/addAddV2 tf.nn.gelu_2/Gelu/add/x:output:0tf.nn.gelu_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
tf.nn.gelu_2/Gelu/mul_1Multf.nn.gelu_2/Gelu/mul:z:0tf.nn.gelu_2/Gelu/add:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 dense_18/StatefulPartitionedCallStatefulPartitionedCalltf.nn.gelu_2/Gelu/mul_1:z:0dense_18_76528dense_18_76530*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_18_layer_call_and_return_conditional_losses_75112ã
dropout_6/PartitionedCallPartitionedCall)dense_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_6_layer_call_and_return_conditional_losses_75123·
tf.__operators__.add_6/AddV2AddV2"dropout_6/PartitionedCall:output:06layer_normalization_5/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ó
flatten/PartitionedCallPartitionedCall tf.__operators__.add_6/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75132þ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_76536dense_76538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75144Ø
softmax/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_softmax_layer_call_and_return_conditional_losses_75155o
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ñ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall!^dense_11/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall!^dense_14/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall!^dense_18/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall.^layer_normalization_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2D
 dense_14/StatefulPartitionedCall dense_14/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2^
-layer_normalization_5/StatefulPartitionedCall-layer_normalization_5/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:

_output_shapes
: :($
"
_output_shapes
:1

Û
%__inference_model_layer_call_fn_77004

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:	

unknown_46:	

unknown_47:	

unknown_48:

unknown_49:	


unknown_50:

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*T
_read_only_resource_inputs6
42	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_76084o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*¤
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: :1: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:

_output_shapes
: :($
"
_output_shapes
:1
ï

P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_74800

inputs3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(q
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:«
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
	keep_dims(T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *½75
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1v
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1f
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
É
ù
B__inference_dense_8_layer_call_and_return_conditional_losses_74488

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ü
`
D__inference_reshape_5_layer_call_and_return_conditional_losses_74892

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs


c
D__inference_dropout_5_layer_call_and_return_conditional_losses_79862

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ê
ú
C__inference_dense_14_layer_call_and_return_conditional_losses_74836

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*²
serving_default
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ;
softmax0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:ö
Ï
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-3
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer_with_weights-4
layer-24
layer-25
layer-26
layer_with_weights-5
layer-27
layer_with_weights-6
layer-28
layer-29
layer_with_weights-7
layer-30
 layer-31
!layer-32
"layer_with_weights-8
"layer-33
#layer_with_weights-9
#layer-34
$layer_with_weights-10
$layer-35
%layer-36
&layer-37
'layer-38
(layer-39
)layer-40
*layer-41
+layer-42
,layer-43
-layer_with_weights-11
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer-49
3layer-50
4layer_with_weights-12
4layer-51
5layer-52
6layer-53
7layer_with_weights-13
7layer-54
8layer_with_weights-14
8layer-55
9layer-56
:layer_with_weights-15
:layer-57
;layer-58
<layer-59
=layer_with_weights-16
=layer-60
>layer_with_weights-17
>layer-61
?layer_with_weights-18
?layer-62
@layer-63
Alayer-64
Blayer-65
Clayer-66
Dlayer-67
Elayer-68
Flayer-69
Glayer-70
Hlayer_with_weights-19
Hlayer-71
Ilayer-72
Jlayer-73
Klayer-74
Llayer-75
Mlayer-76
Nlayer-77
Olayer_with_weights-20
Olayer-78
Player-79
Qlayer-80
Rlayer_with_weights-21
Rlayer-81
Slayer_with_weights-22
Slayer-82
Tlayer-83
Ulayer_with_weights-23
Ulayer-84
Vlayer-85
Wlayer-86
Xlayer-87
Ylayer_with_weights-24
Ylayer-88
Zlayer-89
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_default_save_signature
b
signatures"
_tf_keras_network
"
_tf_keras_input_layer
(
c	keras_api"
_tf_keras_layer
¥
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
(
j	keras_api"
_tf_keras_layer
(
k	keras_api"
_tf_keras_layer
¼
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p_random_generator
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
Ä
saxis
	tgamma
ubeta
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
¿

|kernel
}bias
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
)
¡	keras_api"
_tf_keras_layer
)
¢	keras_api"
_tf_keras_layer
)
£	keras_api"
_tf_keras_layer
)
¤	keras_api"
_tf_keras_layer
)
¥	keras_api"
_tf_keras_layer
«
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¬kernel
	­bias
®	variables
¯trainable_variables
°regularization_losses
±	keras_api
²__call__
+³&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸_random_generator
¹__call__
+º&call_and_return_all_conditional_losses"
_tf_keras_layer
)
»	keras_api"
_tf_keras_layer
Í
	¼axis

½gamma
	¾beta
¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Åkernel
	Æbias
Ç	variables
Ètrainable_variables
Éregularization_losses
Ê	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"
_tf_keras_layer
)
Í	keras_api"
_tf_keras_layer
Ã
Îkernel
	Ïbias
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ó	keras_api
Ô__call__
+Õ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ö	variables
×trainable_variables
Øregularization_losses
Ù	keras_api
Ú_random_generator
Û__call__
+Ü&call_and_return_all_conditional_losses"
_tf_keras_layer
)
Ý	keras_api"
_tf_keras_layer
Í
	Þaxis

ßgamma
	àbeta
á	variables
âtrainable_variables
ãregularization_losses
ä	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
çkernel
	èbias
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
í__call__
+î&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
ïkernel
	ðbias
ñ	variables
òtrainable_variables
óregularization_losses
ô	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"
_tf_keras_layer
«
÷	variables
øtrainable_variables
ùregularization_losses
ú	keras_api
û__call__
+ü&call_and_return_all_conditional_losses"
_tf_keras_layer
)
ý	keras_api"
_tf_keras_layer
)
þ	keras_api"
_tf_keras_layer
)
ÿ	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£_random_generator
¤__call__
+¥&call_and_return_all_conditional_losses"
_tf_keras_layer
)
¦	keras_api"
_tf_keras_layer
Í
	§axis

¨gamma
	©beta
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
°kernel
	±bias
²	variables
³trainable_variables
´regularization_losses
µ	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layer
)
¸	keras_api"
_tf_keras_layer
Ã
¹kernel
	ºbias
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
¿__call__
+À&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å_random_generator
Æ__call__
+Ç&call_and_return_all_conditional_losses"
_tf_keras_layer
)
È	keras_api"
_tf_keras_layer
Í
	Éaxis

Êgamma
	Ëbeta
Ì	variables
Ítrainable_variables
Îregularization_losses
Ï	keras_api
Ð__call__
+Ñ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Òkernel
	Óbias
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Úkernel
	Ûbias
Ü	variables
Ýtrainable_variables
Þregularization_losses
ß	keras_api
à__call__
+á&call_and_return_all_conditional_losses"
_tf_keras_layer
«
â	variables
ãtrainable_variables
äregularization_losses
å	keras_api
æ__call__
+ç&call_and_return_all_conditional_losses"
_tf_keras_layer
)
è	keras_api"
_tf_keras_layer
)
é	keras_api"
_tf_keras_layer
)
ê	keras_api"
_tf_keras_layer
)
ë	keras_api"
_tf_keras_layer
)
ì	keras_api"
_tf_keras_layer
)
í	keras_api"
_tf_keras_layer
)
î	keras_api"
_tf_keras_layer
Ã
ïkernel
	ðbias
ñ	variables
òtrainable_variables
óregularization_losses
ô	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"
_tf_keras_layer
)
÷	keras_api"
_tf_keras_layer
)
ø	keras_api"
_tf_keras_layer
)
ù	keras_api"
_tf_keras_layer
)
ú	keras_api"
_tf_keras_layer
)
û	keras_api"
_tf_keras_layer
«
ü	variables
ýtrainable_variables
þregularization_losses
ÿ	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
Í
	axis

gamma
	beta
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
 	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses"
_tf_keras_layer
)
£	keras_api"
_tf_keras_layer
Ã
¤kernel
	¥bias
¦	variables
§trainable_variables
¨regularization_losses
©	keras_api
ª__call__
+«&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¬	variables
­trainable_variables
®regularization_losses
¯	keras_api
°_random_generator
±__call__
+²&call_and_return_all_conditional_losses"
_tf_keras_layer
)
³	keras_api"
_tf_keras_layer
«
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
ºkernel
	»bias
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Â	variables
Ãtrainable_variables
Äregularization_losses
Å	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"
_tf_keras_layer
Ô
t0
u1
|2
}3
4
5
6
7
¬8
­9
½10
¾11
Å12
Æ13
Î14
Ï15
ß16
à17
ç18
è19
ï20
ð21
22
23
24
25
¨26
©27
°28
±29
¹30
º31
Ê32
Ë33
Ò34
Ó35
Ú36
Û37
ï38
ð39
40
41
42
43
44
45
¤46
¥47
º48
»49"
trackable_list_wrapper
Ô
t0
u1
|2
}3
4
5
6
7
¬8
­9
½10
¾11
Å12
Æ13
Î14
Ï15
ß16
à17
ç18
è19
ï20
ð21
22
23
24
25
¨26
©27
°28
±29
¹30
º31
Ê32
Ë33
Ò34
Ó35
Ú36
Û37
ï38
ð39
40
41
42
43
44
45
¤46
¥47
º48
»49"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
a_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
â2ß
%__inference_model_layer_call_fn_75265
%__inference_model_layer_call_fn_76895
%__inference_model_layer_call_fn_77004
%__inference_model_layer_call_fn_76300À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
@__inference_model_layer_call_and_return_conditional_losses_77818
@__inference_model_layer_call_and_return_conditional_losses_78681
@__inference_model_layer_call_and_return_conditional_losses_76543
@__inference_model_layer_call_and_return_conditional_losses_76786À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
 __inference__wrapped_model_74048input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
Íserving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Înon_trainable_variables
Ïlayers
Ðmetrics
 Ñlayer_regularization_losses
Òlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_reshape_layer_call_fn_78797¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_reshape_layer_call_and_return_conditional_losses_78810¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
l	variables
mtrainable_variables
nregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
'__inference_dropout_layer_call_fn_78815
'__inference_dropout_layer_call_fn_78820´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Â2¿
B__inference_dropout_layer_call_and_return_conditional_losses_78825
B__inference_dropout_layer_call_and_return_conditional_losses_78837´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
':%2layer_normalization/gamma
&:$2layer_normalization/beta
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
Ý2Ú
3__inference_layer_normalization_layer_call_fn_78846¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø2õ
N__inference_layer_normalization_layer_call_and_return_conditional_losses_78868¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :2dense_1/kernel
:2dense_1/bias
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_1_layer_call_fn_78877¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_1_layer_call_and_return_conditional_losses_78907¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :2dense_2/kernel
:2dense_2/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_2_layer_call_fn_78916¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_2_layer_call_and_return_conditional_losses_78946¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_reshape_1_layer_call_fn_78951¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_reshape_1_layer_call_and_return_conditional_losses_78965¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_3_layer_call_fn_78974¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_3_layer_call_and_return_conditional_losses_79004¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_reshape_2_layer_call_fn_79009¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_reshape_2_layer_call_and_return_conditional_losses_79022¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :2dense_4/kernel
:2dense_4/bias
0
¬0
­1"
trackable_list_wrapper
0
¬0
­1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
önon_trainable_variables
÷layers
ømetrics
 ùlayer_regularization_losses
úlayer_metrics
®	variables
¯trainable_variables
°regularization_losses
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_4_layer_call_fn_79031¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_4_layer_call_and_return_conditional_losses_79061¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_1_layer_call_fn_79066
)__inference_dropout_1_layer_call_fn_79071´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_1_layer_call_and_return_conditional_losses_79076
D__inference_dropout_1_layer_call_and_return_conditional_losses_79088´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
"
_generic_user_object
 "
trackable_list_wrapper
):'2layer_normalization_1/gamma
(:&2layer_normalization_1/beta
0
½0
¾1"
trackable_list_wrapper
0
½0
¾1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
ß2Ü
5__inference_layer_normalization_1_layer_call_fn_79097¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_79119¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:	2dense_5/kernel
:2dense_5/bias
0
Å0
Æ1"
trackable_list_wrapper
0
Å0
Æ1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ç	variables
Ètrainable_variables
Éregularization_losses
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_5_layer_call_fn_79128¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_5_layer_call_and_return_conditional_losses_79158¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
!:	2dense_6/kernel
:2dense_6/bias
0
Î0
Ï1"
trackable_list_wrapper
0
Î0
Ï1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ð	variables
Ñtrainable_variables
Òregularization_losses
Ô__call__
+Õ&call_and_return_all_conditional_losses
'Õ"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_6_layer_call_fn_79167¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_6_layer_call_and_return_conditional_losses_79197¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ö	variables
×trainable_variables
Øregularization_losses
Û__call__
+Ü&call_and_return_all_conditional_losses
'Ü"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_2_layer_call_fn_79202
)__inference_dropout_2_layer_call_fn_79207´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_2_layer_call_and_return_conditional_losses_79212
D__inference_dropout_2_layer_call_and_return_conditional_losses_79224´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
"
_generic_user_object
 "
trackable_list_wrapper
):'2layer_normalization_2/gamma
(:&2layer_normalization_2/beta
0
ß0
à1"
trackable_list_wrapper
0
ß0
à1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
á	variables
âtrainable_variables
ãregularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
ß2Ü
5__inference_layer_normalization_2_layer_call_fn_79233¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_79255¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :2dense_7/kernel
:2dense_7/bias
0
ç0
è1"
trackable_list_wrapper
0
ç0
è1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
é	variables
êtrainable_variables
ëregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_7_layer_call_fn_79264¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_7_layer_call_and_return_conditional_losses_79294¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :2dense_8/kernel
:2dense_8/bias
0
ï0
ð1"
trackable_list_wrapper
0
ï0
ð1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
ñ	variables
òtrainable_variables
óregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_8_layer_call_fn_79303¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_8_layer_call_and_return_conditional_losses_79333¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
÷	variables
øtrainable_variables
ùregularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_reshape_3_layer_call_fn_79338¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_reshape_3_layer_call_and_return_conditional_losses_79352¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 :2dense_9/kernel
:2dense_9/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_dense_9_layer_call_fn_79361¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_9_layer_call_and_return_conditional_losses_79391¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_reshape_4_layer_call_fn_79396¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_reshape_4_layer_call_and_return_conditional_losses_79409¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2dense_10/kernel
:2dense_10/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_10_layer_call_fn_79418¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_10_layer_call_and_return_conditional_losses_79448¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
	variables
 trainable_variables
¡regularization_losses
¤__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_3_layer_call_fn_79453
)__inference_dropout_3_layer_call_fn_79458´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_3_layer_call_and_return_conditional_losses_79463
D__inference_dropout_3_layer_call_and_return_conditional_losses_79475´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
"
_generic_user_object
 "
trackable_list_wrapper
):'2layer_normalization_3/gamma
(:&2layer_normalization_3/beta
0
¨0
©1"
trackable_list_wrapper
0
¨0
©1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
ß2Ü
5__inference_layer_normalization_3_layer_call_fn_79484¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_79506¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 	2dense_11/kernel
:2dense_11/bias
0
°0
±1"
trackable_list_wrapper
0
°0
±1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
²	variables
³trainable_variables
´regularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_11_layer_call_fn_79515¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_11_layer_call_and_return_conditional_losses_79545¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
": 	2dense_12/kernel
:2dense_12/bias
0
¹0
º1"
trackable_list_wrapper
0
¹0
º1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
»	variables
¼trainable_variables
½regularization_losses
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_12_layer_call_fn_79554¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_12_layer_call_and_return_conditional_losses_79584¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_4_layer_call_fn_79589
)__inference_dropout_4_layer_call_fn_79594´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_4_layer_call_and_return_conditional_losses_79599
D__inference_dropout_4_layer_call_and_return_conditional_losses_79611´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
"
_generic_user_object
 "
trackable_list_wrapper
):'2layer_normalization_4/gamma
(:&2layer_normalization_4/beta
0
Ê0
Ë1"
trackable_list_wrapper
0
Ê0
Ë1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
Ì	variables
Ítrainable_variables
Îregularization_losses
Ð__call__
+Ñ&call_and_return_all_conditional_losses
'Ñ"call_and_return_conditional_losses"
_generic_user_object
ß2Ü
5__inference_layer_normalization_4_layer_call_fn_79620¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_79642¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2dense_13/kernel
:2dense_13/bias
0
Ò0
Ó1"
trackable_list_wrapper
0
Ò0
Ó1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Õnon_trainable_variables
Ölayers
×metrics
 Ølayer_regularization_losses
Ùlayer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_13_layer_call_fn_79651¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_13_layer_call_and_return_conditional_losses_79681¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2dense_14/kernel
:2dense_14/bias
0
Ú0
Û1"
trackable_list_wrapper
0
Ú0
Û1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
Ü	variables
Ýtrainable_variables
Þregularization_losses
à__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_14_layer_call_fn_79690¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_14_layer_call_and_return_conditional_losses_79720¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
â	variables
ãtrainable_variables
äregularization_losses
æ__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_reshape_5_layer_call_fn_79725¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_reshape_5_layer_call_and_return_conditional_losses_79739¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
!:2dense_15/kernel
:2dense_15/bias
0
ï0
ð1"
trackable_list_wrapper
0
ï0
ð1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
ñ	variables
òtrainable_variables
óregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_15_layer_call_fn_79748¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_15_layer_call_and_return_conditional_losses_79778¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
ü	variables
ýtrainable_variables
þregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_reshape_6_layer_call_fn_79783¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_reshape_6_layer_call_and_return_conditional_losses_79796¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2dense_16/kernel
:2dense_16/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_16_layer_call_fn_79805¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_16_layer_call_and_return_conditional_losses_79835¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_5_layer_call_fn_79840
)__inference_dropout_5_layer_call_fn_79845´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_5_layer_call_and_return_conditional_losses_79850
D__inference_dropout_5_layer_call_and_return_conditional_losses_79862´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
"
_generic_user_object
 "
trackable_list_wrapper
):'2layer_normalization_5/gamma
(:&2layer_normalization_5/beta
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ß2Ü
5__inference_layer_normalization_5_layer_call_fn_79871¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_79893¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 	2dense_17/kernel
:2dense_17/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ýnon_trainable_variables
þlayers
ÿmetrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_17_layer_call_fn_79902¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_17_layer_call_and_return_conditional_losses_79932¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
": 	2dense_18/kernel
:2dense_18/bias
0
¤0
¥1"
trackable_list_wrapper
0
¤0
¥1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¦	variables
§trainable_variables
¨regularization_losses
ª__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_18_layer_call_fn_79941¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_18_layer_call_and_return_conditional_losses_79971¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¬	variables
­trainable_variables
®regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_6_layer_call_fn_79976
)__inference_dropout_6_layer_call_fn_79981´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_6_layer_call_and_return_conditional_losses_79986
D__inference_dropout_6_layer_call_and_return_conditional_losses_79998´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
Ñ2Î
'__inference_flatten_layer_call_fn_80003¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_flatten_layer_call_and_return_conditional_losses_80009¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	
2dense/kernel
:
2
dense/bias
0
º0
»1"
trackable_list_wrapper
0
º0
»1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¼	variables
½trainable_variables
¾regularization_losses
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
Ï2Ì
%__inference_dense_layer_call_fn_80018¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_80028¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Â	variables
Ãtrainable_variables
Äregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
Þ2Û
'__inference_softmax_layer_call_fn_80033¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ù2ö
B__inference_softmax_layer_call_and_return_conditional_losses_80038¯
¦²¢
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaults¢

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
æ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
W86
X87
Y88
Z89"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÊBÇ
#__inference_signature_wrapper_78792input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

total

count
	variables
 	keras_api"
_tf_keras_metric
c

¡total

¢count
£
_fn_kwargs
¤	variables
¥	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¡0
¢1"
trackable_list_wrapper
.
¤	variables"
_generic_user_object
	J
Const
J	
Const_1ø
 __inference__wrapped_model_74048Ód¦§tu|}¬­½¾ÅÆÎÏßàïðçè¨©°±¹ºÊËÚÛÒÓïð¤¥º»8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
softmax!
softmaxÿÿÿÿÿÿÿÿÿ
­
C__inference_dense_10_layer_call_and_return_conditional_losses_79448f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_dense_10_layer_call_fn_79418Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1®
C__inference_dense_11_layer_call_and_return_conditional_losses_79545g°±3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_dense_11_layer_call_fn_79515Z°±3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1®
C__inference_dense_12_layer_call_and_return_conditional_losses_79584g¹º4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_dense_12_layer_call_fn_79554Z¹º4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1­
C__inference_dense_13_layer_call_and_return_conditional_losses_79681fÒÓ3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_dense_13_layer_call_fn_79651YÒÓ3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1­
C__inference_dense_14_layer_call_and_return_conditional_losses_79720fÚÛ3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_dense_14_layer_call_fn_79690YÚÛ3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1­
C__inference_dense_15_layer_call_and_return_conditional_losses_79778fïð3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_dense_15_layer_call_fn_79748Yïð3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1­
C__inference_dense_16_layer_call_and_return_conditional_losses_79835f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_dense_16_layer_call_fn_79805Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1®
C__inference_dense_17_layer_call_and_return_conditional_losses_79932g3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_dense_17_layer_call_fn_79902Z3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1®
C__inference_dense_18_layer_call_and_return_conditional_losses_79971g¤¥4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_dense_18_layer_call_fn_79941Z¤¥4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1ª
B__inference_dense_1_layer_call_and_return_conditional_losses_78907d|}3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_dense_1_layer_call_fn_78877W|}3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1¬
B__inference_dense_2_layer_call_and_return_conditional_losses_78946f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_dense_2_layer_call_fn_78916Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1¬
B__inference_dense_3_layer_call_and_return_conditional_losses_79004f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_dense_3_layer_call_fn_78974Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1¬
B__inference_dense_4_layer_call_and_return_conditional_losses_79061f¬­3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_dense_4_layer_call_fn_79031Y¬­3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1­
B__inference_dense_5_layer_call_and_return_conditional_losses_79158gÅÆ3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_dense_5_layer_call_fn_79128ZÅÆ3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1­
B__inference_dense_6_layer_call_and_return_conditional_losses_79197gÎÏ4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_dense_6_layer_call_fn_79167ZÎÏ4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1¬
B__inference_dense_7_layer_call_and_return_conditional_losses_79294fçè3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_dense_7_layer_call_fn_79264Yçè3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1¬
B__inference_dense_8_layer_call_and_return_conditional_losses_79333fïð3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_dense_8_layer_call_fn_79303Yïð3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1¬
B__inference_dense_9_layer_call_and_return_conditional_losses_79391f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_dense_9_layer_call_fn_79361Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1£
@__inference_dense_layer_call_and_return_conditional_losses_80028_º»0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 {
%__inference_dense_layer_call_fn_80018Rº»0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
¬
D__inference_dropout_1_layer_call_and_return_conditional_losses_79076d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 ¬
D__inference_dropout_1_layer_call_and_return_conditional_losses_79088d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_dropout_1_layer_call_fn_79066W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª "ÿÿÿÿÿÿÿÿÿ1
)__inference_dropout_1_layer_call_fn_79071W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª "ÿÿÿÿÿÿÿÿÿ1¬
D__inference_dropout_2_layer_call_and_return_conditional_losses_79212d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 ¬
D__inference_dropout_2_layer_call_and_return_conditional_losses_79224d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_dropout_2_layer_call_fn_79202W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª "ÿÿÿÿÿÿÿÿÿ1
)__inference_dropout_2_layer_call_fn_79207W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª "ÿÿÿÿÿÿÿÿÿ1¬
D__inference_dropout_3_layer_call_and_return_conditional_losses_79463d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 ¬
D__inference_dropout_3_layer_call_and_return_conditional_losses_79475d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_dropout_3_layer_call_fn_79453W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª "ÿÿÿÿÿÿÿÿÿ1
)__inference_dropout_3_layer_call_fn_79458W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª "ÿÿÿÿÿÿÿÿÿ1¬
D__inference_dropout_4_layer_call_and_return_conditional_losses_79599d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 ¬
D__inference_dropout_4_layer_call_and_return_conditional_losses_79611d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_dropout_4_layer_call_fn_79589W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª "ÿÿÿÿÿÿÿÿÿ1
)__inference_dropout_4_layer_call_fn_79594W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª "ÿÿÿÿÿÿÿÿÿ1¬
D__inference_dropout_5_layer_call_and_return_conditional_losses_79850d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 ¬
D__inference_dropout_5_layer_call_and_return_conditional_losses_79862d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_dropout_5_layer_call_fn_79840W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª "ÿÿÿÿÿÿÿÿÿ1
)__inference_dropout_5_layer_call_fn_79845W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª "ÿÿÿÿÿÿÿÿÿ1¬
D__inference_dropout_6_layer_call_and_return_conditional_losses_79986d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 ¬
D__inference_dropout_6_layer_call_and_return_conditional_losses_79998d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_dropout_6_layer_call_fn_79976W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª "ÿÿÿÿÿÿÿÿÿ1
)__inference_dropout_6_layer_call_fn_79981W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª "ÿÿÿÿÿÿÿÿÿ1ª
B__inference_dropout_layer_call_and_return_conditional_losses_78825d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 ª
B__inference_dropout_layer_call_and_return_conditional_losses_78837d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_dropout_layer_call_fn_78815W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª "ÿÿÿÿÿÿÿÿÿ1
'__inference_dropout_layer_call_fn_78820W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ1
p
ª "ÿÿÿÿÿÿÿÿÿ1£
B__inference_flatten_layer_call_and_return_conditional_losses_80009]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
'__inference_flatten_layer_call_fn_80003P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿº
P__inference_layer_normalization_1_layer_call_and_return_conditional_losses_79119f½¾3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
5__inference_layer_normalization_1_layer_call_fn_79097Y½¾3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1º
P__inference_layer_normalization_2_layer_call_and_return_conditional_losses_79255fßà3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
5__inference_layer_normalization_2_layer_call_fn_79233Yßà3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1º
P__inference_layer_normalization_3_layer_call_and_return_conditional_losses_79506f¨©3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
5__inference_layer_normalization_3_layer_call_fn_79484Y¨©3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1º
P__inference_layer_normalization_4_layer_call_and_return_conditional_losses_79642fÊË3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
5__inference_layer_normalization_4_layer_call_fn_79620YÊË3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1º
P__inference_layer_normalization_5_layer_call_and_return_conditional_losses_79893f3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
5__inference_layer_normalization_5_layer_call_fn_79871Y3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1¶
N__inference_layer_normalization_layer_call_and_return_conditional_losses_78868dtu3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
3__inference_layer_normalization_layer_call_fn_78846Wtu3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1
@__inference_model_layer_call_and_return_conditional_losses_76543Ïd¦§tu|}¬­½¾ÅÆÎÏßàïðçè¨©°±¹ºÊËÚÛÒÓïð¤¥º»@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
@__inference_model_layer_call_and_return_conditional_losses_76786Ïd¦§tu|}¬­½¾ÅÆÎÏßàïðçè¨©°±¹ºÊËÚÛÒÓïð¤¥º»@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
@__inference_model_layer_call_and_return_conditional_losses_77818Îd¦§tu|}¬­½¾ÅÆÎÏßàïðçè¨©°±¹ºÊËÚÛÒÓïð¤¥º»?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
@__inference_model_layer_call_and_return_conditional_losses_78681Îd¦§tu|}¬­½¾ÅÆÎÏßàïðçè¨©°±¹ºÊËÚÛÒÓïð¤¥º»?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ì
%__inference_model_layer_call_fn_75265Âd¦§tu|}¬­½¾ÅÆÎÏßàïðçè¨©°±¹ºÊËÚÛÒÓïð¤¥º»@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
ì
%__inference_model_layer_call_fn_76300Âd¦§tu|}¬­½¾ÅÆÎÏßàïðçè¨©°±¹ºÊËÚÛÒÓïð¤¥º»@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
ë
%__inference_model_layer_call_fn_76895Ád¦§tu|}¬­½¾ÅÆÎÏßàïðçè¨©°±¹ºÊËÚÛÒÓïð¤¥º»?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
ë
%__inference_model_layer_call_fn_77004Ád¦§tu|}¬­½¾ÅÆÎÏßàïðçè¨©°±¹ºÊËÚÛÒÓïð¤¥º»?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
¬
D__inference_reshape_1_layer_call_and_return_conditional_losses_78965d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_reshape_1_layer_call_fn_78951W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª " ÿÿÿÿÿÿÿÿÿ1¬
D__inference_reshape_2_layer_call_and_return_conditional_losses_79022d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_reshape_2_layer_call_fn_79009W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1¬
D__inference_reshape_3_layer_call_and_return_conditional_losses_79352d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_reshape_3_layer_call_fn_79338W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª " ÿÿÿÿÿÿÿÿÿ1¬
D__inference_reshape_4_layer_call_and_return_conditional_losses_79409d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_reshape_4_layer_call_fn_79396W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1¬
D__inference_reshape_5_layer_call_and_return_conditional_losses_79739d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_reshape_5_layer_call_fn_79725W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ1
ª " ÿÿÿÿÿÿÿÿÿ1¬
D__inference_reshape_6_layer_call_and_return_conditional_losses_79796d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_reshape_6_layer_call_fn_79783W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª "ÿÿÿÿÿÿÿÿÿ1ª
B__inference_reshape_layer_call_and_return_conditional_losses_78810d7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ1
 
'__inference_reshape_layer_call_fn_78797W7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ1
#__inference_signature_wrapper_78792Þd¦§tu|}¬­½¾ÅÆÎÏßàïðçè¨©°±¹ºÊËÚÛÒÓïð¤¥º»C¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ"1ª.
,
softmax!
softmaxÿÿÿÿÿÿÿÿÿ
¢
B__inference_softmax_layer_call_and_return_conditional_losses_80038\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ


 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 z
'__inference_softmax_layer_call_fn_80033O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ


 
ª "ÿÿÿÿÿÿÿÿÿ
