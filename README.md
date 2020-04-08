# HDRImage-ToneMapping

**組員:林尚箴  學號:B10502214**

專案介紹:透過演算法將不一樣曝光時間的照片合成成一張 HDR(High Dynamic Range)Image，之後HDR Image重新Mapping回LDR

此專案分為以下步驟

**1.Taking Phtotos:**  
使用手機拍一組不一樣曝光時間的照片

**2.Image sampling:**  
在不同曝光時間的照片中選擇大約100組同一位置的Sample點用來完成HDR的計算，這邊一開始使用Random的方式，後來改用Uniform的方式Sample

**3.Radiance Map:**  
透過Debevec's所提供的演算法分別解出三個通道的Radiance Curve(g 函數),並透過Radiance Curve 算出HDR Image

**4.Tone Mapping:**  
採用Reinhard的演算法將HDR Image 映射回LDR
