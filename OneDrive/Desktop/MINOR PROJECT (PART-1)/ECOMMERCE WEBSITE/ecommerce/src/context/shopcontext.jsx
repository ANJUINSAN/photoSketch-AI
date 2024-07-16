import { createContext, useState } from "react";
import all_product from "../components/images/Assets/all_product";

export const ShopContext = createContext(null);

const getdefaultcart = () => {
  let cart = {};
  for (let index = 0; index < all_product.length + 1; index++) {
    cart[index] = 0;
  }
  return cart;
};
const ShopContextProvider = (props) => {
  const [cartItem, setcartItem] = useState(getdefaultcart());

  // console.log(cartItem)

  const addToCart = (itemId) => {
    setcartItem((prev) => ({ ...prev, [itemId]: prev[itemId] + 1 }));
    // console.log(cartItem);
  };
  const removeFromCart = (itemId) => {
    setcartItem((prev) => ({ ...prev, [itemId]: prev[itemId] - 1 }));
    // console.log(cartItem);
  };
  const getTotalCartAmount = () => {
    let totalAmount = 0;
    for (const item in cartItem) {
      if (cartItem[item] > 0) {
        let itemInfo = all_product.find(
          product=> product.id === Number(item)
        );

        totalAmount += itemInfo.new_price * cartItem[item];
        // console.log(totalAmount)
      }
    }
    return totalAmount;
  };
  const getTotalCartItems = () => {
    let totalItem = 0;

    for (const item in cartItem) {
    //   console.log(typeof item);
      if (cartItem[item] > 0) {
        totalItem += cartItem[item];
      }
    }
    return totalItem;
  };
  const contextVal = {getTotalCartItems,getTotalCartAmount, all_product, cartItem,addToCart, removeFromCart};
  return (
    <ShopContext.Provider value={contextVal}>
      {props.children}
    </ShopContext.Provider>
  );
};
export default ShopContextProvider;
