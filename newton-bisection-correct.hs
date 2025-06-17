import Data.Time.Clock
import Data.List (sort)

tolerance :: Double
tolerance = 1e-6

maxItr :: Int
maxItr = 50

maxDepth :: Int
maxDepth = 10000

--fold_r :: (a -> a -> a) -> a -> [a] -> a
--fold_r _ x []     = x
--fold_r f x (t:ts) = fold_r f (x `f` t) ts
--
--zipp :: [a] -> [b] -> [(a,b)]
--zipp [] _ = []
--zipp _ [] = []
--zipp (x:xs) (y:ys) = (x, y) : zipp xs ys
--
--mapp :: (a -> b) -> [a] -> [b]
--mapp f []     = []
--mapp f (x:xs) = (f x) : mapp f xs
--
--revLst :: [a] -> [a]
--revLst l = rev l []
--    where
--        rev [] l     = l
--        rev (x:xs) l = rev xs (x:l)

readCoefficients :: FilePath -> IO [Double]
readCoefficients path = do
  content <- readFile path
  return $ map read $ lines content

fujiwaraBound :: [Double] -> Double
fujiwaraBound (a_n:coeffs) =
    let n = length coeffs
        ks = [1..fromIntegral n]
        bs = zipWith (\a k -> (abs (a / a_n)) ** (1.0 / k)) coeffs ks
    in 2.0 * maximum bs

normalizeByMaxCoefficient :: [Double] -> [Double]
normalizeByMaxCoefficient coeffs =
  let maxCoeff = maximum (map abs coeffs)
  in if maxCoeff /= 0 then map (/ maxCoeff) coeffs else coeffs

polyVal :: [Double] -> Double -> Double
polyVal coeffs x =
  sum $ map (\(a,b) -> a*b) (zip coeffs (reverse $ map (x **) [0..fromIntegral (length coeffs - 1)]))

polyValSign :: [Double] -> Double -> Double
polyValSign p x
  | abs x <= 1 = signum $ polyVal p x
  | otherwise =
      let y = 1 / x
          n = length p - 1
          reverseP = reverse p
      in if even n then signum $ polyVal reverseP y
      else signum $ polyVal reverseP y * signum y

polyFraction :: [Double] -> [Double] -> Double -> Double
polyFraction p q x =
  let n = length p - length q
  in if abs x > 1
     then let lead = x ^^ n
              y = 1 / x
              pr = reverse p
              qr = reverse q
          in lead * polyVal pr y / polyVal qr y
     else polyVal p x / polyVal q x

polyDerivative :: [Double] -> [Double]
polyDerivative p =
    let n = length p - 1
    in map (\(a, b) -> a * b) (zip (map fromIntegral [n, n-1..1]) p)

newtonRaphson :: [Double] -> [Double] -> Double -> Double -> Double -> Int -> Maybe Double
newtonRaphson p dp x0 a b 0 = Nothing
newtonRaphson p dp x0 a b iteration =
  let ratio = polyFraction p dp x0
      x1 = x0 - ratio
  in if abs (x1 - x0) < tolerance then Just x1
     else if not (a <= x1 && x1 <= b) then Nothing
     else newtonRaphson p dp x1 a b (iteration - 1)

bisection :: [Double] -> Double -> Double -> (Double, Bool)
bisection p a b
  | aSign == 0.0 = (a, True)
  | bSign == 0.0 = (b, True)
  | abs (b - a) < tolerance = ((a + b) / 2, False)
  | midSign == 0.0 = (midpoint, False)
  | aSign * midSign < 0 = bisection p a midpoint
  | otherwise = bisection p midpoint b
  where
    aSign = polyValSign p a
    bSign = polyValSign p b
    midpoint = (a + b) / 2
    midSign = polyValSign p midpoint


findRoots :: [Double] -> Double -> Double -> Int -> [Double]
findRoots coeffs a b depth
  | depth > maxDepth = []
  | degree == 0 = []
  | degree == 1 =
      let [a1, a0] = coeffs
      in if a1 == 0 then [] else
         let root = -a0 / a1
         in [root | a <= root, root <= b]
  | otherwise =
       let dp_coeffs = polyDerivative coeffs
           normalized_d_coeffs = normalizeByMaxCoefficient dp_coeffs
           d_roots = findRoots normalized_d_coeffs a b (depth + 1)
           allPoints = sort (a : b : d_roots)
           checkInterval x1 x2 acc =
             let p_low  = polyValSign coeffs x1
                 p_high = polyValSign coeffs x2
             in if p_low == 0.0
                   then x1 : acc
                else if p_high == 0.0
                   then x2 : acc
                else if p_low * p_high < 0
                   then
                     let initialGuess = (x1 + x2) / 2
                     in case newtonRaphson coeffs dp_coeffs initialGuess x1 x2 maxItr of
                        Just root -> root : acc
                        Nothing ->
                             let (point, isRoot) = bisection coeffs x1 x2
                             in if isRoot
                                    then point : acc
                                    else
                                       case newtonRaphson coeffs dp_coeffs point x1 x2 maxItr of
                                         Just root -> root : acc
                                         Nothing   -> acc
                else acc
       in foldr (\(x1, x2) acc -> checkInterval x1 x2 acc)
            [] (zip allPoints (tail allPoints))

  where
    degree = length coeffs - 1

main = do
  p <- readCoefficients "poly_coeff_newton.csv"

  start <- getCurrentTime
  let bound = fujiwaraBound p
  let roots = findRoots p (-bound) bound 0
  putStrLn $ "Newton-Raphson & Bisection real roots: " ++ show roots
  end <- getCurrentTime

  putStrLn $ "Time taken: " ++ show (diffUTCTime end start)
