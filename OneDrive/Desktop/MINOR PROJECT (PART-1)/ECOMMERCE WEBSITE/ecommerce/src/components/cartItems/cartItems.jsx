import React, { useContext } from "react";
import "./cartItems.css";
import { ShopContext } from "../../context/shopcontext";
import remove_icon from "../images/Assets/cart_cross_icon.png";
import { Link } from "react-router-dom";

export default function cartItems() {
  const { getTotalCartAmount, all_product, cartItem, removeFromCart } =
    useContext(ShopContext);

  return (
    <div className="cartItems">
      <div className="cartItems-format-main">
        <p>Products</p>
        <p>Title</p>
        <p>Price</p>
        <p>Quantity</p>
        <p>Total</p>
        <p>Remove</p>
      </div>
      <hr />

      {all_product.map((e) => {
        if (cartItem[e.id] > 0) {
          return (
            <div>
              <div className="cartItems-format">
                <img src={e.image} alt="" className="carticon-product-icon" />
                <p>{e.name}</p>
                <p>${e.new_price}</p>
                <button className="cartItems-quantity">{cartItem[e.id]}</button>
                <p>${e.new_price * cartItem[e.id]}</p>
                <img
                  className="cartItems-remove-icon"
                  onClick={() => {
                    removeFromCart(e.id);
                  }}
                  src={remove_icon}
                  alt=""
                />
              </div>
              <hr />
            </div>
          );
        }
        return null;
      })}
      <div className="cartItems-down">
        <div className="cartItems-total">
          <h1>Cart Totals</h1>
          <div>
            <div className="cartItems-total-item">
              <p>subtotal</p>
              <p>${getTotalCartAmount()}</p>
            </div>
            <hr />
            <div className="cartItems-total-item">
              <p>shipping fee</p>
              <p>FREE</p>
            </div>
            <hr />
            <div className="cartItems-total-item">
              <h3>Total</h3>
              <h3>${getTotalCartAmount()}</h3>
            </div>
          </div>

          <Link to='/payments' style={{ textDecoration: "none" }} >
            <button>PROCEED TO CHECKOUT</button>
          </Link>
        </div>
        <div className="cartItems-promocode">
          <p>If you have a promo code, enter it here</p>
          <div className="cartItems-promobox">
            <input type="text" name="" id="" placeholder="promo code" />
            <button className="btn-promo">submit</button>
          </div>
        </div>
      </div>
    </div>
  );
}
