# Valószínűségszámítás és a Bayes-tétel

## 1. Feltételes valószínűség és a Bayes-tétel
A Bayes-tétel a feltételes valószínűségek kiszámítására és a megfigyelt bizonyítékok (evidenciák) alapján történő valószínűség-frissítésre szolgáló alapvető matematikai tétel.

### A Bayes-tétel matematikai képlete
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Ahol a tagok jelentése:
* **$P(A|B)$ (Poszterior valószínűség):** Annak a valószínűsége, hogy az $A$ hipotézis igaz, miután megfigyeltük a $B$ bizonyítékot / eseményt.
* **$P(A)$ (Prior vagy előzetes valószínűség):** Az $A$ hipotézis valószínűsége a bizonyíték megismerése előtt (alapgyakoriság / base rate).
* **$P(B|A)$ (Likelihood / Valószerűség):** Annak a valószínűsége, hogy a $B$ bizonyítékot megfigyeljük, feltéve, hogy az $A$ hipotézis igaz.
* **$P(B)$ (Teljes valószínűség / Evidencia):** A $B$ bizonyíték bekövetkezésének marginális valószínűsége az összes lehetséges állapot felett.

### A teljes valószínűség tétele a nevezőben
Ha $A$ komplementere $\neg A$ (tehát $A$ nem következik be):
$$P(B) = P(B|A) \cdot P(A) + P(B|\neg A) \cdot P(\neg A)$$

Ezért a részletes Bayes-képlet:
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|\neg A) \cdot P(\neg A)}$$

---

## 2. Mintapélda: Orvosi diagnosztikai szűrés (Alapszint-figyelmen kívül hagyása)

### A feladat adatai:
Egy ritka betegség szűrésére orvosi tesztet végeznek egy lakosságon.
1. **Betegség előfordulási gyakorisága (Prior):** A népesség 1%-a beteg:
   * $P(\text{Beteg}) = 0.01$
   * $P(\text{Egészséges}) = 0.99$
2. **A teszt megbízhatósága (Likelihoodok):**
   * **Szenzitivitás (True Positive Rate):** Ha valaki beteg, a teszt 99%-os eséllyel pozitív:  
     $P(+ \mid \text{Beteg}) = 0.99$
   * **Fals pozitív arány (False Positive Rate):** Ha valaki egészséges, a teszt 5%-os eséllyel tévesen pozitívat jelez:  
     $P(+ \mid \text{Egészséges}) = 0.05$ (Specificitás: 95%, azaz $P(- \mid \text{Egészséges}) = 0.95$)

### A kérdés:
Ha egy véletlenszerűen kiválasztott személy tesztje **pozitív** lett, mennyi a valószínűsége, hogy valóban beteg? Vagyis mennyi a $P(\text{Beteg} \mid +)$ poszterior valószínűség?

### Lépésről lépésre történő levezetés:
1. **Számláló kiszámítása:**
   $$\text{Számláló} = P(+ \mid \text{Beteg}) \cdot P(\text{Beteg}) = 0.99 \cdot 0.01 = 0.0099$$
2. **Nevező kiszámítása (Teljes valószínűség):**
   $$P(+) = [P(+ \mid \text{Beteg}) \cdot P(\text{Beteg})] + [P(+ \mid \text{Egészséges}) \cdot P(\text{Egészséges})]$$
   $$P(+) = (0.99 \cdot 0.01) + (0.05 \cdot 0.99) = 0.0099 + 0.0495 = 0.0594$$
3. **A végeredmény meghatározása:**
   $$P(\text{Beteg} \mid +) = \frac{0.0099}{0.0594} = \frac{1}{6} \approx 0.1667 \quad (\mathbf{16.67\%})$$

### Pedagógiai és döntési konklúzió (Base Rate Fallacy)
Bár a teszt látszólag 99%-os pontosságú, egy pozitív lelet esetén a páciens valójában csak kb. **16.67%** eséllyel beteg. Ennek oka az alacsony alapgyakoriság (prior = 1%): az egészséges emberek 5%-os tévedési aránya abszolút számban jóval több téves riasztást generál, mint ahány valódi beteg van a populációban.